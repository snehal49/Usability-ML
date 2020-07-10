With an increase in the number of mobile apps making their way to users, there is a growing need for tools to support the app design process. While many tools focus on increasing the pace of development, few attempt to aide the designer in generating more creative solutions. In this work, we take creativity as the combination of novelty and utility. Particularly during development of user interfaces, assessment of utility (primarily usability) is iterative, rigorous, and time-consuming. The objective of the proposed work is to explore and evaluate the use of machine learning to predict usability measures for mobile app interfaces as a means to automate usability evaluations. Specifically, a convolutional neural network (CNN) is used to accurately (nearly 90%) predict three usability measures: regularity, complexity, and touchability. This tool automates the assessment of utility in app design, freeing up the designer to seek de- signs that are novel and thus creative.

Google Colab was used to run and evaluate the model. 

Dataset files:
1) weather app images zip folder contains 205 weather app screenshots
2) weather app annotated images zip folder contains 410 weather app annotated screenshots (with data augmentation)
3) measures for weather app_410.npy contains usability measure values for 410 weather app annotated screenshots 
