# ifnude - nudity detection that just works

A neural nets powered library that can detect nudity in images of both real humans and drawings. It takes an image as input and tells you exactly what NSFW parts of the body are visible. Optionally, you can censor the said parts.

<img src="https://i.imgur.com/0KPJbl9.jpg" width=600>

### Installation
```bash
pip install ifnude
```

### Example
> Note: Importing ifnude for the first time will download a 139MB module to "/your/home/dir/.ifnude/", just once.
```python
from ifnude import detect

# use mode="fast" for x3 speed with slightly lower accuracy
print(detect('/path/to/nsfw.png'))
```

Instead of a path, you can use a variable that contains an image loaded through `cv2` (opencv) or `PIL` (pillow).

#### Output
```
[
  {'box': [164, 188, 246, 271], 'score': 0.8253238201141357, 'label': 'EXPOSED_BREAST_F'},
  {'box': [252, 190, 335, 270], 'score': 0.8235630989074707, 'label': 'EXPOSED_BREAST_F'}
]
```

### Censor Video

```python

from ifnude import detector

detector.censor_video(inp_video_path='path/to/video.mp4', out_video_path='path/to/output_video.mp4') #make sure to use forward slash (/)

```

### Output

##### raw video

https://streamable.com/q8jp8c

##### censored video

https://streamable.com/su3yko


### Credits
This is fork of [NudeNet](https://pypi.org/project/NudeNet/) library which doesn't work anymore. 


