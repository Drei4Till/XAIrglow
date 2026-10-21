# XAIrglow
In atmospheric physics and spatio-temporal forecasting, deep learning models are used in predicting physical hyperparameters (e.g., wavelength, amplitude, angle, time-period) from simulated sequences of OH airglow waves. While these networks are effective at regression tasks, they often act as "black boxes." When a network predicts multiple physical parameters simultaneously from a single superimposed wave image, it is unclear what spatial and temporal features it actually utilizes for each specific parameter. Understanding this is vital to ensure the network is learning true physical properties rather than exploiting statistical artifacts in the simulation data.

This project focuses on adapting Explainable AI (XAI) techniques, which are traditionally used for classification, to a multi-output regression context. Attribution methods such as Grad-CAM or Integrated Gradients will be applied to a network trained on simulated wave images with controlled dynamics. The primary objective is to investigate whether the network spatially disentangles the physics—for example, evaluating if the model specifically focuses on the distance between wave crests to determine the wavelength, while analyzing the global orientation of the image to determine the propagation angle.

# Literature: 

[1] Selvaraju, R. R. et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization." ICCV (2017): https://arxiv.org/abs/1610.02391 

[2] Sundararajan, M. et al. "Axiomatic Attribution for Deep Networks." PMLR (2017): https://arxiv.org/abs/1703.01365
