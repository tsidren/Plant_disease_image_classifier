# Plant_disease_image_classifier
## Image_model_disease (TensorFlow)

Using TensorFlow to create an image classification for plant diseases in Python.

## Packages used in the Project 
1. TensorFlow
2. numpy
3. matplotlib
4. fast API
5. uvicorn
6. IO
7. PIL

   
## Initial Setup 
### 1. Numpy package 
to convert image byte data to an array 
```
conda install numpy
```
### 2. Tensorflow package
for creating models 
```
conda install tensorflow
```
### 3. MatPlotLib package
for visualizing model 
```
conda install matplotlib
```
### 4. FastAPI package
for creating web API 
```
conda install fastapi
```
### 5. uvicorn package
for creating ASGI web server implementation in Python
```
pip install uvicorn
```
### 6. IO package
to deal with the bytes data 
```
pip install Python-IO
```
### 7. PIL package
for cleaning the CSV file data 
```
pip install pillow
```
### 9. install npm in the "frontend" directory. [^1]
```
npm install 
```


> [!IMPORTANT]
> 1. If that didn't work then you can try these pip [commands](https://pip.pypa.io/en/stable/user_guide/).
> 2. If that didn't work then you can try these conda [commands](https://www.tutorialspoint.com/how-do-i-install-python-packages-in-anaconda).
> 3. If there is still a problem then you can install through a .whl file for that particular [package](https://www.w3docs.com/snippets/python/how-do-i-install-a-python-package-with-a-whl-file.html).

## Test Drive 
### 1. API initialization 
a. Run the file "step_6.py" in the 'api' directory.
### 2. Answer Fetching
#### a. Postman 
- Open the "Postman" application.
- Select the "POST" method.
- Use this URL to send the data "http://localhost:8000/predict". (The port can be changed by you)
- Select "File" as the 'Key' and name it "file" and Select the image of your choice as 'Value' in the 'Body' label. [^2]
- It should display the Class of the Disease and the Confidence in the answer.
  like this
    ```
    {
    "class": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "confidence": 0.9990307092666626
    }
    ```
#### b. FastAPI docs
- Open the docs page for your API "http://localhost:8000/docs".
- Expand the "POST" tab.
- Click "Try it out" and Select the image of your choice and the "File" input. [^2]
- Click "Execute"
- It should display the Class of the Disease and the Confidence in the answer. (as a response)
  like this 
    ```
    {
      "class": "Apple___Apple_scab", 
      "confidence": 0.9999998807907104
    }
    ```
#### c. Frontend (website)
- Start the website.
```
npm run start
```
- drag and drop the image of your choice [^2]
- It should display the Class of the Disease and the Confidence in the answer. (as a response)

> [!NOTE]
> 1. Remember to perform the initial setup before test-driving.
> 2. Your API program should always run when test-driving.
> 3. Your testing and training machine should be the same.
>    > NVIDIA Trained model will not run on the AMD machine 

[^1]: I had to do this for running the website. in the "Frontend" directory {where npm is installed}  ```set NODE_OPTIONS=--openssl-legacy-provider```. if this doesn't work then manually change  ```"start": "react-scripts start"```  to   ```"start": "react-scripts --openssl-legacy-provider start"```   In your "package.json" file

[^2]: I would suggest that you use an image from the "DATA/test/Apple___Apple_scab" directory :sweat_smile:
The model is still weak and it cannot identify complex images :sweat_smile::sweat_smile:
