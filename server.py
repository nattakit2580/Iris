#import model # Import the python file containing the ML model
from flask import Flask, request, render_template,jsonify # Import flask libraries
import pickle as pk
import numpy as np

# Initialize the flask class and specify the templates directory
app = Flask(__name__,template_folder="templates")

# Load the model
loaded_model = pk.load(open('model.pkl','rb'))

# Dictionary containing the mapping
variety_mappings = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Function for classification based on inputs
def classify(a, b, c, d):
    print('\n classifier is called \n')
    arr = np.array([a, b, c, d]) # Convert to numpy array
    arr = arr.astype(np.float64) # Change the data type to float
    query = arr.reshape(1, -1) # Reshape the array
    prediction = variety_mappings[loaded_model.predict(query)[0]] # Retrieve from dictionary
    return prediction # Return the prediction

# Default route set as 'home'
@app.route('/home')
def home():
    return render_template('home.html') # Render home.html

# Route 'classify' accepts GET request
@app.route('/classify',methods=['POST','GET'])
def classify_type():
    try:
        sepal_len = request.args.get('slen') # Get parameters for sepal length
        sepal_wid = request.args.get('swid') # Get parameters for sepal width
        petal_len = request.args.get('plen') # Get parameters for petal length
        petal_wid = request.args.get('pwid') # Get parameters for petal width

        # Get the output from the classification model
        #variety = model.classify(sepal_len, sepal_wid, petal_len, petal_wid)
        variety = classify(sepal_len, sepal_wid, petal_len, petal_wid)

        # Render the output in new HTML page
        return render_template('output.html', variety=variety)
    except:
        return 'Error'

# Run the Flask server
if(__name__=='__main__'):
    app.run(debug=True)        
