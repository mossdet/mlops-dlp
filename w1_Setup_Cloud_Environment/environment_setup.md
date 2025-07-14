# Setup ssh connection
## 1. Save public key to ~/.ssh/
## 2. The keys need to be read-writable only by me, so change the read, write, execute rights:
	chmod 600 ~/.ssh/mlops-zc.pem
## 3. Connect to instance:
	ssh -i ~/.ssh/mlops-zc.pem ubuntu@3.144.47.7
## 4. Optionally, set up the connection in ssh:
	nano ~/.ssh/config
	Host mlops-zc
	        HostName 3.144.47.74
	        User ubuntu
	        IdentityFile ~/.ssh/mlops-zc.pem
	        StrictHostKeyChecking no
## 5. log in using ssh mlops-zc



# Install UV Python
	wget -qO- https://astral.sh/uv/install.sh | sh

# Install Docker
	- sudo apt update
	- sudo apt install docker.io

	- Add user to docker group to avoid having to run Docker as sudo every time:
		- sudo groupadd docker
		- sudo usermod -aG docker $USER
		- log out and log back in

# Install Docker Compose
	
	- Download:
		wget https://github.com/docker/compose/releases/download/v2.37.0/docker-compose-linux-x86_64 -O docker-compose

	- Make file executable:
		chmod +x docker-compose

	- Since docker compose was downloaded to the path: "~/Software/docker-compose", we need to add this path to the system variables so that it is accesible from everywhere:
		> nano .bashrc
		> at the end of the file, add the following:
		export PATH="${HOME}/Software:${PATH}"


