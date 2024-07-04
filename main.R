# Load required libraries
library(tidyverse)
library(plotly)
library(caret)
library(FactoMineR)
library(factoextra)

# Read the dataset
df <- read.csv('/path/....penguins_Size.csv')

# Make a copy of the dataset
dfc <- df

# Inspect the dataset
head(df)
summary(df)
str(df)
df %>% select_if(is.character) %>% map(~ table(.))

# Handle missing values
df <- df %>% mutate(sex = ifelse(sex == '.', NA, sex))
df[336, ]
sum(is.na(df))
df %>% filter_all(any_vars(is.na(.)))

# Impute missing values and encode categorical variables
df <- df %>% 
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric)

preProcess_missingdata_model <- preProcess(df, method='knnImpute')
df <- predict(preProcess_missingdata_model, newdata = df)

# Normalize the data
preProcess_range_model <- preProcess(df, method='range')
df_sca <- predict(preProcess_range_model, newdata = df)

# Impute missing values in the scaled data
new_df <- predict(preProcess_missingdata_model, newdata = df_sca)
sum(is.na(new_df))

# Round and map values
new_df <- new_df %>%
  mutate(sex = round(sex),
         sex = ifelse(sex == 0, 'Female', 'Male'),
         species = factor(species, levels = c(0, 1, 2), labels = c('Adeile', 'Chinstrap', 'Gentoo')),
         island = factor(island, levels = c(0, 1, 2), labels = c('Biscoe', 'Dream', 'Torgersen')))

# Bar plot
plot_ly(data = new_df, x = ~island, type = 'bar', color = ~species,
        colors = c('Adeile' = 'rgb(251,117,4)', 'Chinstrap' = 'rgb(167,98,188)', 'Gentoo' = 'rgb(4,115,116)'),
        width = 1200, height = 900) %>%
  layout(facet = ~species)

# Scatter plot: flipper length vs body mass
plot_ly(data = new_df, x = ~flipper_length_mm, y = ~body_mass_g, color = ~species,
        colors = c('Adeile' = 'rgb(251,117,4)', 'Chinstrap' = 'rgb(167,98,188)', 'Gentoo' = 'rgb(4,115,116)'),
        symbol = ~species, symbols = c('Adeile' = 'circle', 'Chinstrap' = 'triangle-up', 'Gentoo' = 'square'),
        height = 760) %>%
  layout(title = 'flipper length (mm) vs body mass (g)')

# Scatter plot: culmen length vs culmen depth
plot_ly(data = new_df, x = ~culmen_length_mm, y = ~culmen_depth_mm, color = ~species,
        colors = c('Adeile' = 'rgb(251,117,4)', 'Chinstrap' = 'rgb(167,98,188)', 'Gentoo' = 'rgb(4,115,116)'),
        symbol = ~species, symbols = c('Adeile' = 'circle', 'Chinstrap' = 'triangle-up', 'Gentoo' = 'square'),
        height = 760) %>%
  layout(title = 'culmen length (mm) vs culmen depth (mm)')

# Scatter plot: flipper length vs body mass by species and sex
plot_ly(data = new_df, x = ~flipper_length_mm, y = ~body_mass_g, color = ~sex,
        colors = c('Male' = 'darkblue', 'Female' = 'deeppink'), facet = ~species) %>%
  layout(title = 'Species based Gender scatter plot', showlegend = FALSE, height = 800)

# Violin plots
plot_ly() %>%
  add_trace(data = new_df, type = 'violin', x = ~species, y = ~body_mass_g, name = 'Body Mass (g)',
            box = list(visible = TRUE), meanline = list(visible = TRUE), points = 'all') %>%
  add_trace(data = new_df, type = 'violin', x = ~species, y = ~flipper_length_mm, name = 'Flipper Length (mm)',
            box = list(visible = TRUE), meanline = list(visible = TRUE), points = 'all') %>%
  layout(title = 'Violin Plots', showlegend = FALSE)

# PCA analysis
pca <- PCA(new_df[, c('culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g')], scale.unit = TRUE, ncp = 4)
pca_df <- data.frame(pca$ind$coord)
pca_df$species <- new_df$species

# Scatter plot: PCA
plot_ly(data = pca_df, x = ~Dim.1, y = ~Dim.2, color = ~species,
        colors = c('Adeile' = 'rgb(251,117,4)', 'Chinstrap' = 'rgb(167,98,188)', 'Gentoo' = 'rgb(4,115,116)'),
        symbol = ~species, symbols = c('Adeile' = 'circle', 'Chinstrap' = 'triangle-up', 'Gentoo' = 'square')) %>%
  layout(title = 'Principal component 1 vs Principal component 2')

# Bar plots: PCA loadings
loadings <- as.data.frame(pca$var$coord)
plot_ly() %>%
  add_trace(type = 'bar', x = ~loadings$Dim.1, y = rownames(loadings), name = 'PC1', orientation = 'h') %>%
  add_trace(type = 'bar', x = ~loadings$Dim.2, y = rownames(loadings), name = 'PC2', orientation = 'h') %>%
  add_trace(type = 'bar', x = ~loadings$Dim.3, y = rownames(loadings), name = 'PC3', orientation = 'h') %>%
  add_trace(type = 'bar', x = ~loadings$Dim.4, y = rownames(loadings), name = 'PC4', orientation = 'h') %>%
  layout(title = 'PCA Loadings', barmode = 'group')
