pipeline{
    agent any

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
    }
}