pipeline{
    agent any

    environment {

        VENV_DIR = 'venv'
    }


    stages {
        stage('Cloning Github Repo to Jenkins')
        {
            steps {
                script{

                    echo 'Cloning Github Repo to Jenkins ... '
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token-for-jenkins', url: 'https://github.com/OmarAymanHassan/Hotel-Reservation-Prediction.git']])
                }
            }
        }



        stage('Setting Up our virtual environment and installing dependencies')
        {
            steps {
                script{

                    echo 'Setting Up our virtual environment and installing dependencies ... '
                    sh '''

                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .

                    '''
                }
            }
        }
    }
}