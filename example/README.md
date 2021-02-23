# ManimalAI Examples

## hello_manimal.py

Minimal example with a random action agent.

![frame0](../docs/images/frame0.png)



## capture.py
Captures all of the arena configuration setting with the bird's eye view.

![2-2-3](../docs/images/captures/2-2-3.png)

![6-20-3](../docs/images/captures/6-20-3.png)


## manual_control.py

Manual agent control example.

This example uses `pygame` library and threre is a compatibility problem with latest `pygame` and `rodentia` libraries on Linux. So please use `pygame 1.9.6` on Linux.

### MacOSX

```
$ pip3 install pygame==2.0.1
$ python3 manual_control.py
```

### Ubuntu

```
$ pip3 install pygame==1.9.6
$ python3 manual_control.py
```


![eight_arm_maze0](../docs/images/eight_arm_maze0.gif)

