pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                git 'https://github.com/suup1/ml-ci-project.git'
            }
        }

        stage('Verify Files') {
            steps {
                bat 'dir'
                bat 'dir data'
            }
        }

        stage('Setup Virtual Environment') {
            steps {
                bat 'python -m venv venv'
                bat 'venv\\Scripts\\activate && pip install -r requirements.txt'
            }
        }

        stage('Train Model') {
            steps {
                bat 'venv\\Scripts\\activate && python src/train.py'
            }
        }

        stage('Archive Model') {
            steps {
                archiveArtifacts artifacts: 'models/*.pkl', fingerprint: true
            }
        }
    }
}