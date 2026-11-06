
pipeline {
    agent any

    options {
        timestamps()
    }

    environment {
        IMAGE_NAME = 'wheat-disease-classifier'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    python --version
                    python -m pip install --upgrade pip

                    python -m pip install \
                        --index-url https://download.pytorch.org/whl/cpu \
                        "torch>=2.1" "torchvision>=0.16"

                    python -m pip install \
                        Pillow numpy pytest flake8
                '''
            }
        }

        stage('Lint') {
            steps {
                sh 'flake8 wheat_classifier tests'
            }
        }

        stage('Test') {
            steps {
                sh 'pytest -v'
            }
        }

        stage('Docker Build') {
            steps {
                sh '''
                    docker build \
                        -t ${IMAGE_NAME}:${BUILD_NUMBER} \
                        -t ${IMAGE_NAME}:latest .
                '''
            }
        }
    }
}
