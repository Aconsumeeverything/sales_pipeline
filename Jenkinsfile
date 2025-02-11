pipeline {
    agent any  // Run on any available Jenkins agent

    stages {
        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/Aconsumeeverything/sales_pipeline.git'

            }
        }

        stage('Clean Data') {
            steps {
                sh 'docker build -t data_cleaning .'  // Build Docker image
                sh 'docker run --rm -v $(pwd):/app data_cleaning'  // Run data cleaning
            }
        }

        stage('Train Model') {
            steps {
                sh 'python train.py'  // Run the training script
            }
        }

        stage('Save Artifacts') {
            steps {
                archiveArtifacts artifacts: 'model.pkl, logs/*', fingerprint: true
            }
        }
    }
}
