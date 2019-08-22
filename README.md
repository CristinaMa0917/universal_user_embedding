# universal_user_embedding
Intern work in Alibaba: Generate a universal user embedding from the past search queries of that user. The universal user embedding is able to reflect the user's long time interest and short time preference.

Universal user embedding helps the generation of person profile and selection of target people for advertising.

Original design is as follows. 

![images](https://github.com/CristinaMa0917/universal_user_embedding/blob/master/images/img1.png)

Freeze the text embedding from pretrained BERT and concat with the downstream text embedding (like ELMO):

![images](https://github.com/CristinaMa0917/universal_user_embedding/blob/master/images/img2.png)

Change the max pooling into pretrained bert for text embedding generation.

![images](https://github.com/CristinaMa0917/universal_user_embedding/blob/master/images/img3.png)

