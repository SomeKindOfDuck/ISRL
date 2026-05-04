# 全パラメータの結果でincentive learningが生じるか比較
library(tidyverse)
library(tools)
library(legendry)

font_config <- theme(
  axis.text = element_text(size = 10),
  axis.title = element_text(size = 12),
  legend.text = element_text(size = 6),
  legend.title = element_text(size = 6),
  strip.text = element_text(color = "white"),
  strip.background = element_rect(color = "black", fill = "black")
)


merged_data <- list.files("./data/inctv", full.names = T, pattern = "csv") %>%
  lapply(., function(path) {
    d <- read.csv(path)
    idenifyers <- str_split(file_path_sans_ext(basename(path)), "-", simplify = T)
    level <- ifelse(idenifyers[2] == "control", "NoPre", "Pre")
    condition <- ifelse(idenifyers[1] == "high2low", "High incentive", "Low incentive")
    d %>%
      mutate(
        condition = factor(condition, levels = c("High incentive", "Low incentive")),
        level = factor(level, levels = c("Pre", "NoPre"))
      ) %>%
      filter(homeostatic_point <= 120)
}) %>%
  do.call(rbind, .)

# ホメオスタシス状態に基づいてHighとLow状態を定義して、状態ごとにphi_oを平均する
state_data <- merged_data %>%
  mutate(
    state = case_when(
      homeostatic_point < 60 ~ "High",
      TRUE ~ "Low"
    ),
    state = factor(state, levels = c("High", "Low"))
  ) %>%
  group_by(homeostatic_kernel, covert_kernel, kappa, rho, condition, level, state) %>%
  summarise(mean_phi_o = mean(phi_o), .groups = "drop")

# x軸をNestして表示するために整形してggplotに流す
dev.new(width = 4, height = 3)

state_data$kappa %>% unique
state_data$covert_kernel %>% unique

state_data %>%
  mutate(
    state = factor(state, levels = c("High", "Low")),
    level = factor(level, levels = c("Pre", "NoPre")),
    x_lab = interaction(level, state, sep = ",")
  ) %>%
  filter(homeostatic_kernel == 6, covert_kernel == 9) %>%
  ggplot(.) +
    geom_boxplot(
      aes(
        x = x_lab,
        y = mean_phi_o
      )
    ) +
    scale_x_discrete(
      limits = as.vector(outer(
        c("Pre", "NoPre"),
        c("High", "Low"),
        paste,
        sep = ","
      )),
      guide = legendry::guide_axis_nested(key_range_auto(sep = ","))
    ) +
    ylab("Response strength (φₒ)") +
    xlab("Pre-exposure × Test deprivation level") +
    facet_wrap(~condition) +
    theme_classic() +
    theme(aspect.ratio = 1.) +
    font_config +
    facet_wrap(~kappa~condition)

ggsave("./figs/inctv/main_result.jpg", dpi = 300, units = "cm")

# 条件ごとにstate間の差分を算出
diff_data <- state_data %>%
  pivot_wider(
    names_from = state,
    values_from = mean_phi_o
  ) %>%
  mutate(
    phi_o_diff = High - Low
  )

ggplot(diff_data) +
  geom_boxplot(aes(x = level, y = phi_o_diff)) +
  ylab("φₒ difference between\nhigh and low deprivation levels") +
  xlab("Pre-exposure") +
  facet_grid(~condition) +
  theme_classic() +
  theme(aspect.ratio = 1.5) +
  font_config

ggsave("./figs/inctv/diff_phi_o.jpg", dpi = 300, units = "cm")

ggplot(merged_data) +
  stat_summary(
    geom = "line",
    fun = "mean",
    aes(
      x = homeostatic_point,
      y = phi_o,
      color = level,
      group = level
    )
) +
  facet_wrap(~condition)
