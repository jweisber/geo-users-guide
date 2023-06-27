library(tidyverse)

df <- read_csv("50-agents.csv", show_col_types = FALSE) %>% 
  group_by(bias, boost, dist) %>%
  summarize(mean_score = mean(score), .groups = "drop") %>%
  group_by(bias) %>%
  mutate(is_best = mean_score == min(mean_score))

p_labeller <- function(string) paste("italic(p) ==", string)

ggplot(df) +
  facet_wrap(vars(bias), ncol = 2, labeller = as_labeller(p_labeller, label_parsed)) +
  geom_tile(aes(x = dist, y = boost, fill = mean_score)) +
  geom_point(data = df %>% filter(is_best), aes(x = dist, y = boost), shape = 8, colour = "white") +
  geom_vline(xintercept = seq(-.05, 2.05, .1), colour = "grey", alpha = .75) + 
  geom_hline(yintercept = seq(-.05, 1.05, .1), colour = "grey", alpha = .75) +
  scale_fill_gradientn(colours = rev(rainbow(4)), name = "Brier score\n") +
  scale_x_continuous(breaks = seq(0, 2, by = .1), expand = c(0, 0)) +  
  scale_y_continuous(breaks = seq(0, 1, by = .1), expand = c(0, 0)) +
  labs(x = expression(italic(ϵ)), y = expression(italic(c))) +
  theme_minimal() +
  theme(panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.spacing = unit(1.5, "lines"))

ggsave("fig-4.png", dpi = 300, width = 10, height = 8)
