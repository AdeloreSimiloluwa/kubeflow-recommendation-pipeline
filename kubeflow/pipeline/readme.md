# Building a Kubeflow Pipeline.
For this example, we would be working on a supervised learning classification problem using an Artificial Neural network. The goal is to determine if a customer would churn/leave or not.

### Step 1:
The first step is to make sure you have successfully deployed kubeflow, all its dependencies and you have access to the dashboard. There are many ways to deploy kubeflow. It can be deployed on your system(MacOs, Windows, Linux) and on the cloud(GCP, AWS, IBM, Azure). The following links should guide you on how to deploy kubeflow on different platforms:

    1. Deploy Kubeflow on Windows
    2. Deploy Kubeflow on MacOs
    3. Deploy Kubeflow on Linux
    4. Deploy Kubeflow on GCP
    5. Deploy Kubeflow on AWS
    6. Deploy Kubeflow on IBM
    7. Deploy Kubeflow on Azure
    
For this example we would be using GCP to build the pipeline. To hasten the process we have put together the following steps for deploying Kubeflow on GCP.

    1. Create a new project 
    2. Enable boost shell on cloud shell
    3. Enable deployment manager API
    4. Setup OAuth credentials
    5. Setup environmental files and enable other api’s
          1. export DEPLOYMENT_NAME=kf-codelab
          2. export PROJECT_ID=your-project-id
          3. export ZONE=us-central1-c or us-east1-c
          4. gcloud config set project ${PROJECT_ID}
          5. gcloud config set compute/zone ${ZONE}
          gcloud services enable   cloudresourcemanager.googleapis.com   iam.googleapis.com   file.googleapis.com   ml.googleapis.com

    6. Download kfctl v1.0 on the cloud shell with  
    wget https://github.com/kubeflow/kfctl/releases/download/v1.1.0/kfctl_v1.1.0-0-g9a3621e_linux.tar.gz
    7. Extract the downloaded package with tar -xvf kfctl_v1.1.0-0-g9a3621e_linux.tar.gz
    8. Export kfctl path using export PATH=$PATH:/home/home-name/
    go to home directory using cd
    Get the current directory with pwd
    9. Authorize gcloud with gcloud auth login and gcloud auth application-default login
    10.set config_uri 
     export   CONFIG_URI="https://raw.githubusercontent.com/kubeflow/manifests/v1.0-branch/kfdef/kfctl_gcp_iap.v1.0.2.yaml"

    11. Set up oauth secret 
    export CLIENT_ID=your-auth-client-id
    export CLIENT_SECRET=your-oauth-client-secret

    12. Setup kubeflow environment variables
    KF_NAME=kf-codelab
    BASE_DIR=${HOME}/kf_deployments
    export KF_DIR=${BASE_DIR}/${KF_NAME}

    13. Apply Kubeflow 
    mkdir -p ${KF_DIR}
    cd ${KF_DIR}
    kfctl apply -V -f ${CONFIG_URI}

    14. Create Cluster
    gcloud container clusters get-credentials ${KF_NAME} --zone ${ZONE} --project ${PROJECT_ID}
    kubectl -n kubeflow get all
    kubectl -n istio-system get ingress

### Step 2:
Once you have followed the steps and deployed kubeflow on GCP, paste the link given in your browser to get access to the Kubeflow dashboard. 


Start by setting up a jupyter notebook through the Notebook Servers tab. Follow the steps here to set up your notebook  server.



### Step 3:
Once the server is set up, click connect for access to your jupyter notebook. Open up the terminal and clone the following repository from GitHub.



$ git clone https://github.com/AdeloreSimiloluwa/Artificial-Neural-Network
Open the file “Artificial Neural Networks Pipeline”. The data needed for this project has already been uploaded on git so you can proceed to run each cell in the notebook. Start by installing the needed libraries and restart the kernel. The next step is to import the data and preprocess.


### Step 4:
Now installing the kubeflow pipeline SDK. The kubeflow pipeline SDK offers python packages that you can use to run your machine learning workflows. To install SDK, run this cell and restart your kernel.

!pip install -q kfp --upgrade --user


### Step 5:
Create python functions for training, testing and prediction and convert them into containers with this function, func to container op.
Training function:

Predict function:


Now wrap the up into container components with:

name = comp.func_to_container_op(func)

Parameters :    func is the python function you intend to convert

### Step 6:
Usually, when defining the kubflow pipeline, one would have to manually manipulate a YAML file, but with Kubeflow Pipelines SDK you can define your pipeline with :

@dsl.pipeline(name= “”, description = “”)
Before this, you need to initialize a kubeflow client that would enable communication with the Pipelines API server so you can create runs and experiment from your jupyter notebook.

client = kfp.Client()

After this, you define the parameters that should go into the pipeline. In this case we first define the data path, the model path(where the model is stored), and the index of the test data you want predicted.


### Step 7:
Now we can define the pipeline components, compile and run it on the dashboard. With ContainerOp func we would define the components, order of operations  and dependencies of the pipeline.

Here we create the training components and attach persistent volumes to be mounted to the container.

Now print your results:


ContainerOp parameters include:

name - the name displayed for the component execution during runtime.
image - image tag for the Docker container to be used.
pvolumes - dictionary of paths and associated Persistent Volumes to be mounted to the container before execution.
arguments - command to be run by the container at runtime.

Run the next two cells to compile the pipeline and run it within an experiment. Click run to view your pipeline on the Kubeflow pipeline UI.

The components you defined in the notebook should be displayed on the UI.
Once the components are done running you can check the logs for your prediction result. It would display the Churn rate for the index you provide, the level of confidence and the actual label for that index.

