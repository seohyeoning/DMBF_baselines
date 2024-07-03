## Baseline Models
We compared the performance of the various models with that of our proposed method, DMBF, as described below.

- **Deep learning-based unimodal models**:  
  This category includes EEGNet[^1], PSD-VIMSNet[^2], and EEG-based spatial-temporal CNN (ESTCNN)[^3]. VIMSNet and ESTCNN are models designed for motion sickness and fatigue detection, respectively, using EEG data. Following the methodology of a previous study[^2], we extracted features from the EEG data using power spectral density (PSD) before inputting them into the models. 

- **Machine learning-based unimodal models**:  
  This category includes PSD-K-nearest neighbor (KNN), PSD-support vector machine (SVM), PSD-Gaussian Naive Bayes (GNB), and common spatial pattern (CSP)-linear discriminant analysis (LDA). Following the method used by Zheng et al.[^4], features were extracted using PSD and classified with KNN, SVM, and GNB. CSP with LDA[^5] was also used for comparison.

- **Static fusion models**:  
  This category includes early fusion (concatenation), mid fusion (concatenation, summation), late fusion (concatenation), and Attention-TFN[^6]. Based on a study on biosignal fusion[^7], the early, mid, and late fusion methods were used as representative baselines, performing fusion at the signal, feature, and decision levels, respectively. All these models used EEGNet as the backbone network. DMBF was also compared with Attention-TFN, which is a recent multi-biosignal fusion model for motion sickness detection.
    
- **Dynamic fusion models**:  
  This category includes attention-based adaptive fusion (ABAF)[^8], multimodal dynamics (mmDynamics)[^9], and mmDynamics with EEGNet. We compared DMBF with the latest dynamic fusion models ABAF and mmDynamics. For a fair comparison, we included a version of mmDynamics with EEGNet as the backbone network, named "mmDynamics with EEGNet."


[^1]: Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of neural engineering*, 15(5), 056013.
[^2]: Liu, R., Wang, Y., & Sourina, O. (2022). VIMSNet: An effective network for visually induced motion sickness detection. *Signal, Image and Video Processing*, 16(8), 2029-2036.
[^3]: Gao, Z., Yu, S., Yang, Z., & Zhang, Y. (2019). EEG-based spatio-temporal convolutional neural network for driver fatigue evaluation. *IEEE Transactions on Neural Networks and Learning Systems*, 30(9), 2755-2763.
[^4]: Zheng, Y., Qiu, Y., & Liu, W. (2022). A new feature selection approach for driving fatigue EEG detection with a modified machine learning algorithm. *Computers in Biology and Medicine*, 147, 105718.
[^5]: Blankertz, B., Dornhege, G., Krauledat, M., Müller, K. R., & Curio, G. (2007). The non-invasive Berlin brain–computer interface: Fast acquisition of effective performance in untrained subjects. *NeuroImage*, 37(2), 539-550.
[^6]: Hwang, J. U., Bang, J. S., & Lee, S. W. (2022). Classification of motion sickness levels using multimodal biosignals in real driving conditions. In *2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC)* (pp. 1304-1309). IEEE.
[^7]: Haghighat, M., Abdel-Mottaleb, M., & Alhalabi, W. (2016). Discriminant correlation analysis: Real-time feature level fusion for multimodal biometric recognition. *IEEE Transactions on Information Forensics and Security*, 11(9), 1984-1996.
[^8]: Li, S., Wang, Y., Chen, Z., & Hu, B. (2022). Adaptive multimodal fusion with attention guided deep supervision net for grading hepatocellular carcinoma. *IEEE Journal of Biomedical and Health Informatics*, 26(8), 4123-4131.
[^9]: Han, Z., Yang, F., Huang, J., Zhang, C., & Yao, J. (2022). Multimodal dynamics: Dynamical fusion for trustworthy multimodal classification. In *2022 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 20707-20717).





