# replicate_t2c_fixest.R
# T2c 主设定的 R/fixest 独立复核 (投稿前必做)
# 对应 Python: redesign_analysis_v8.py 的 T2c heat_rigidity 连续交互
# 期望: beta2 ≈ +0.253 (cluster p ≈ 0.005, wild bootstrap p ≈ 0.007)
#
# 运行前准备 (一次性):
#   install.packages(c("fixest", "wildboottestjlr", "data.table"))
# 用法: Rscript replicate_t2c_fixest.R

.libPaths('D:/Project/01_科研项目/论文3_RPS消纳_RPS/.rlibs')
library(data.table)
library(fixest)

# wildboottestjlr/fwildclusterboot 在 R 4.6.1 尚无 CRAN 构建,
# 改用 pairs cluster bootstrap (base R 实现, 999次) 做独立 p 值交叉验证。

BASE <- "D:/Project/01_科研项目/论文3_RPS消纳_RPS"
df <- fread(file.path(BASE, "master_panel_v8d.csv"))

# ---- 样本与变量 (与 redesign_analysis_v8.py 完全一致) ----
d <- df[year >= 2018 & year <= 2024 & !province_en %in% c("Tibet", "Xinjiang")]
setorder(d, province_en, year)
d[, dweight := nonhydro_weight_binding - shift(nonhydro_weight_binding), by = province_en]

# heat_rigidity: 2023 基期, 供热容量(万千瓦*10) / 火电装机(MW), 无供暖=结构零
p23 <- df[year == 2023, .(province_en, yb_heat_supply_capacity_mw, installed_capacity_mw_thermal)]
p23[, heat_rigidity := fifelse(is.na(yb_heat_supply_capacity_mw), 0,
                               yb_heat_supply_capacity_mw * 10 / installed_capacity_mw_thermal)]
d <- merge(d, p23[, .(province_en, heat_rigidity)], by = "province_en", all.x = TRUE)
d[, heat_z := (heat_rigidity - mean(heat_rigidity, na.rm = TRUE)) / sd(heat_rigidity, na.rm = TRUE)]

# T2c 样本: 2019-2024 (v8d补齐2020控制变量后连续)
t2 <- d[year >= 2019 & year <= 2024]
t2 <- t2[complete.cases(t2[, .(wind_curtailment_pct, dweight, heat_z,
                               wind_cap_growth_pct, solar_cap_growth_pct,
                               consumption_growth_pct)])]
cat("T2c sample: n =", nrow(t2), ", provinces =", uniqueN(t2$province_en), "\n")

# ---- 主设定: wind ~ dweight*heat_z + controls | province + year ----
m <- feols(wind_curtailment_pct ~ dweight:heat_z + dweight +
             wind_cap_growth_pct + solar_cap_growth_pct + consumption_growth_pct |
             province_en + year,
           data = t2, cluster = ~province_en)
cat("\n== T2c main specification (fixest) ==\n")
print(summary(m))
b2 <- coef(m)["dweight:heat_z"]
p2 <- pvalue(m)["dweight:heat_z"]
cat(sprintf("\nbeta2 = %.6f (Python: 0.229357), cluster p = %.4f (Python: 0.0002)\n", b2, p2))

# ---- heat x year FE 稳健性 (Python: beta2 = 0.367, p = 0.010) ----
m2 <- feols(wind_curtailment_pct ~ dweight:heat_z + dweight +
              wind_cap_growth_pct + solar_cap_growth_pct + consumption_growth_pct +
              heat_z:factor(year) |
              province_en + year,
            data = t2, cluster = ~province_en)
cat("\n== T2c + heat x year FE ==\n")
print(summary(m2))

# ---- pairs cluster bootstrap (999次, 省级重抽样) ----
set.seed(20260724)
provs <- unique(t2$province_en)
boot_b <- replicate(999, {
  ps <- sample(provs, replace = TRUE)
  bd <- rbindlist(lapply(seq_along(ps), function(i) {
    z <- t2[province_en == ps[i]]; z$province_en <- paste0(ps[i], "_", i); z
  }))
  mb <- tryCatch(feols(wind_curtailment_pct ~ dweight:heat_z + dweight +
                         wind_cap_growth_pct + solar_cap_growth_pct + consumption_growth_pct |
                         province_en + year,
                       data = bd, cluster = ~province_en),
                 error = function(e) NULL)
  if (is.null(mb)) return(NA_real_)
  unname(coef(mb)["dweight:heat_z"])
})
boot_b <- boot_b[is.finite(boot_b)]
ci <- quantile(boot_b, c(0.025, 0.975))
cat(sprintf("\n== pairs cluster bootstrap (percentile 95%% CI for beta2) ==\nCI = [%.3f, %.3f], excludes 0: %s\n",
            ci[1], ci[2], ifelse(ci[1] > 0, "YES (p<0.05)", "no")))

# ---- T1 互证设定: dt_post x heat_z (2018-2024) ----
t1 <- d[!is.na(delta_target_i)]
t1[, post := as.integer(year >= 2024)]
t1[, dt_post := delta_target_i * post]
t1 <- t1[complete.cases(t1[, .(wind_curtailment_pct, dt_post, heat_z,
                               wind_cap_growth_pct, solar_cap_growth_pct,
                               consumption_growth_pct)])]
m1 <- feols(wind_curtailment_pct ~ dt_post:heat_z + dt_post +
              wind_cap_growth_pct + solar_cap_growth_pct + consumption_growth_pct |
              province_en + year,
            data = t1, cluster = ~province_en)
cat("\n== T1 corroboration (n =", nrow(t1), ") ==\n")
print(summary(m1))
cat(sprintf("T1 beta2 = %.6f (Python: 0.094718), cluster p = %.4f (Python: 0.0307)\n",
            coef(m1)["dt_post:heat_z"], pvalue(m1)["dt_post:heat_z"]))

cat("\n复核完成。容差建议: |beta 差| < 0.01, p 值同显著性级别即视为通过。\n")
