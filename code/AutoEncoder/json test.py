a = '''{
  "team": "red",
  "datetime" : "2021-11-08 17:39:21" ,
  "sensor":{
		"body":[[-0.9830701,0.03186613,0.3214496,-0.01367188,0.02734375,0.046875,32.875,-10.75,0.3214496,-1.38855,0.9277344,1.803589,-24.88534,64.23254,-161.9948]],
		"left":[[-0.9830701,0.03186613,0.3214496,-0.01367188,0.02734375,0.046875,32.875,-10.75,0.3214496,-1.38855,0.9277344,1.803589,-24.88534,64.23254,-161.9948]],
		"right":[[-0.9830701,0.03186613,0.3214496,-0.01367188,0.02734375,0.046875,32.875,-10.75,0.3214496,-1.38855,0.9277344,1.803589,-24.88534,64.23254,-161.9948]],
		"pelvis":[[-0.9830701,0.03186613,0.3214496,-0.01367188,0.02734375,0.046875,32.875,-10.75,0.3214496,-1.38855,0.9277344,1.803589,-24.88534,64.23254,-161.9948]]
        }
    }
'''

import json
import numpy as np
jj = json.loads(a)
jj = jj['sensor']
body = np.array(jj['body'])
print(body)

pelvis = np.array(jj['pelvis'])
print(pelvis)

left = np.array(jj['left'])
print(left)

right = np.array(jj['right'])
print(right)
concat = np.concatenate((body,pelvis,left,right),1)
print("result",concat,concat.shape)