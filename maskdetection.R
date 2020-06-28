## Author: Gyan Shashwat
## MasK detection system for covid-19

library(keras)
library(EBImage)
library(stringr)
library(pbapply)

secondImage <- readImage("train/mask.1.jpg")
display(secondImage)
# Set image size
width <- 100
height <- 100

extract_feature <- function(dir_path, width, height, labelsExist = T) {
  img_size <- width * height
  
  ## List images in path
  images_names <- list.files(dir_path)
  
  if(labelsExist){
    ## Select only mask or nonmask images
    masknomask <- str_extract(images_names, "^(mask|nonmask)")
    # Set mask == 0 and nonmask == 1
    key <- c("mask" = 0, "nonmask" = 1)
    y <- key[masknomask]
  }
  
  print(paste("Start processing", length(images_names), "images"))
  ## This function will resize an image, turn it into greyscale
  feature_list <- pblapply(images_names, function(imgname) {
    ## Read image
    img <- readImage(file.path(dir_path, imgname))
    ## Resize image
    img_resized <- resize(img, w = width, h = height)
    ## Set to grayscale (normalized to max)
    grayimg <- channel(img_resized, "gray")
    ## Get the image as a matrix
    img_matrix <- grayimg@.Data
    ## Coerce to a vector (row-wise)
    img_vector <- as.vector(t(img_matrix))
    return(img_vector)
  })
  ## bind the list of vector into matrix
  feature_matrix <- do.call(rbind, feature_list)
  feature_matrix <- as.data.frame(feature_matrix)
  ## Set names
  names(feature_matrix) <- paste0("pixel", c(1:img_size))
  
  if(labelsExist){
    return(list(X = feature_matrix, y = y))
  }else{
    return(feature_matrix)
  }
}


# Takes approx. 15min
trainData <- extract_feature("train", width, height)
# Takes slightly less
testData <- extract_feature("test", width, height, labelsExist = F)

# Check processing on second image
par(mar = rep(0, 4))
testimage <- t(matrix(as.numeric(trainData$X[2,]),
                    nrow = width, ncol = height, T))
image(t(apply(testimage, 2, rev)), col = gray.colors(12),
      axes = F)

# Save / load
save(trainData, testData, file = "maskDetection.RData")
# load("maskDetection.RData")


# Fix structure for 2d CNN
train_array <- t(trainData$X)
dim(train_array) <- c(100, 100, nrow(trainData$X), 1)
# Reorder dimensions
train_array <- aperm(train_array, c(3,1,2,4))

test_array <- t(testData)
dim(test_array) <- c(100, 100, nrow(testData), 1)
# Reorder dimensions
test_array <- aperm(test_array, c(3,1,2,4))

# Check image again
testimage <- train_array[2,,,]
image(t(apply(testimage, 2, rev)), col = gray.colors(12),
      axes = F)
library(tensorflow)
#install_tensorflow()
# Build CNN model
model <- keras_model_sequential() 
model %>% 
  layer_conv_2d(kernel_size = c(3, 3), filter = 32,
                activation = "relu", padding = "same",
                input_shape = c(100, 100, 1),
                data_format = "channels_last") %>%
  layer_conv_2d(kernel_size = c(3, 3), filter = 32,
                activation = "relu", padding = "valid") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_dropout(rate = 0.25) %>%
  
  layer_conv_2d(kernel_size = c(3, 3), filter = 64, strides = 2,
                activation = "relu", padding = "same") %>%
  layer_conv_2d(kernel_size = c(3, 3), filter = 64,
                activation = "relu", padding = "valid") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_dropout(rate = 0.25) %>%
  
  layer_flatten() %>%
  layer_dense(units = 100, activation = "relu") %>% 
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = "adam",
  metrics = c('accuracy')
)

history <- model %>% fit(
  x = train_array, y = as.numeric(trainData$y), 
  epochs = 50, batch_size = 100, 
  validation_split = 0.2
)

plot(history)
model %>% save_model_hdf5("CNN100newmodel.h5")
new_model <- load_model_hdf5("CNN100newmodel.h5")
summary(new_model)
# Compute probabilities and predictions on test set
predictions <-  predict_classes(new_model, test_array)
probabilities <- predict_proba(new_model, test_array)

# Visual inspection of 32 cases
set.seed(100)
random <- sample(1:nrow(testData), 32)
preds <- predictions[random,]
probs <- as.vector(round(probabilities[random,], 2))

par(mfrow = c(4, 8), mar = rep(0, 4))
for(i in 1:length(random)){
  image(t(apply(test_array[random[i],,,], 2, rev)),
        col = gray.colors(12), axes = F)
  legend("topright", legend = ifelse(preds[i] == 0, "masked", "nonmasked"),
         text.col = ifelse(preds[i] == 0, 2, 4), bty = "n", text.font = 2)
  legend("topleft", legend = probs[i], bty = "n", col = "white")
}
