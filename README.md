# Voice-Recognition

Uses an unsupervised approach using GMM and SVM.

Extracts the MFCC and PLP features and assesses the accuracy of both.

The person can then record a new dialogue and the algorithm gives the following accuracies:

| Model  | Feature Extraction Method | Accuracy on Test Set (%) | 
| ------------- | ------------- | ------------- |
| SVM  | MFCC  | 72.29 |
| SVM  | PLP  | 51.88 |
| GMM  | MFCC  | 82.5 |
| GMM  | PLP  | 84.79 |
