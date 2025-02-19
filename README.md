Name: 劉正德 / Liu Cheng-Te

## Structure:
1. Model: Pre-trained ResNet-50 with a modified final layer for 50 classes
2. Augmentation: Extensive data augmentation with custom noise and transforms
3. Optimizer: Adam with learning rate scheduling
4. Evaluation: Validation accuracy and loss calculated after every epoch
5. Submission: Generates submission.csv for predictions

## Process:
1. **Data Preparation**:
    Data Augmentation:
        Applied transformations include flipping, rotation, color jitter, noise addition (Gaussian, Poisson, Speckle, Salt and Pepper)
    Normalization:
        Used ImageNet mean and standard deviation for normalization
    Dataset Expansion:
        Combined original dataset with additional augmented dataset using ConcatDataset
2. **Training**:
    Setup:
        Epochs: 13
        Batch size: 32
        Optimizer: Adam
        Loss function: CrossEntropyLoss
        Learning rate schedule:
            Epochs 1–3: 0.001
            Epochs 4–7: 0.0005
            Epochs 8–10: 0.0001
            Epochs 11–13: 0.00001
    Validation:
        Stratified sampling splits 10% data for validation
3. **Model Architecture**:
    Pre-trained ResNet-50:
        All layers trainable
        Final fully connected layer replaced with nn.Linear(in_features, 50)
4. **Evaluation**:
    Validation loss and accuracy computed after every epoch
    Final model saved as resnet50_final_weights.pth
5. **Testing and Submission**:
    Test images processed with normalization and resized to fit ResNet-50 input dimensions

## Discussion:
This training focused on image recognition. Initially, I used a CNN, but its performance was poor. Later, I applied transfer learning as mentioned in class and selected ResNet-50 as the base model, which eventually yielded acceptable results. To further improve performance, I believe I should apply more data augmentation to the dataset. A unique aspect of this attempt was that I manually adjusted the learning rate by observing the learning curve instead of using a scheduler.
