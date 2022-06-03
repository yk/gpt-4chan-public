clone https://github.com/kingoflolz/mesh-transformer-jax and put this code inside

`model` is from https://github.com/okbuddyhololive/project-cybertard with slight changes

(you only need to do the above things if you want to run jax inference. for hugging face, it is not necessary)

then run `uvicorn --host 0.0.0.0 --port 8080 serve_api:app`

I use python 3.9.12 and install requirements.txt, then uninstall jax, jaxlib, tensorflow, and tensorflow-cpu

install `jax==0.2.12 jaxlib==0.1.67 tensorflow==2.5.0 markupsafe==2.0.1 uvicorn fastapi loguru`
