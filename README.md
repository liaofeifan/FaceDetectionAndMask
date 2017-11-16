# mask

## introduction
This fun-mask project is mainly based on opencv, user can drag mask to face, and draw your own masks.

we only support one person by far.

For face detection, haar cascade classifier in opencv is directly used. 

for finger detection, the procedure is skin color detection, erosion-dilation, median filter. Finger detection is stable under normal light condition with no skin-color-like-background.

## prerequisit
```
opencv 3.3.1
python 2.7.14 
numpy 1.13.3
scipy 1.0.0
scikit-image 0.13.1
scikit-learn 0.19.1
```

## usage
Preprepared masks are displayed at top left region. You can drag the mask from exhibition region to you mask. If you want to change the mask, just drag another one to your face.
"Click" on the second yellow button, than you can enter into "Paper", draw the mask on the top left region with red bound. When you finish, click on the third yellow button, than this self-mask will be displayed at the third mask. 
Just drag it to you face!
