## 环境准备
Linux
```sh
  virtualenv --system-site-packages -p python3.4 tensorflow_apps
  pip install Django==1.11.7
  pip install --upgrade tensorflow
```
Or mac
```sh
  pyvenv tensorflow_apps
  pip install Django==1.11.7
  pip install --upgrade tensorflow
```


## Docker
```sh
  docker pull registry.docker-cn.com/understate/tensorflow-apps:1.0
  docker run -d -p 8000:8000 registry.docker-cn.com/understate/tensorflow-apps:1.0
```

##  测试
```sh
  curl -l -H "Content-type: application/json" -X POST -d '{"x_data": ["2017-09-22 04:50:00","2017-09-22 09:14:00","2017-10-24 08:10:00"],"y_data": [9.7000000000,12.3000000000,10.9000000000]}' http://localhost:8000/curve_fitting
```