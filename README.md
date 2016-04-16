#face2movie
detect faces in pictures and makes movie/gif out of it

##why i wrote this
i started taking pictures two years ago with my webcam every hour. It was very funny just to see what results you get. I thought why not make a video out of it. But with only having your face in diffenrent locations, is not very nice in a movie.
This is why i wrote this programm. Just for making nice movies of myself:D

##install
for the face-detection it uses [OpenCV](http://opencv.org/), so you have to install it with python bindings

##how it works
+ detecting the face (using OpenCV)
+ then croping the face
+ detecting 2 eyes (also OpenCV) in the face
+ calculate the movement, zoom rate and rotation of the picture
+ apply all to the image
+ append the image to the movie file

##usage
```
Usage: face2movie.py [options]

Options:
  -h, --help            show this help message and exit
  -i IMAGEFOLDER, --imagefolder=IMAGEFOLDER
                        Path of images
  -s FACESCALE, --facescale=FACESCALE
                        scale of the face (default is 1/3)
  -f FPS, --fps=FPS     fps of the resulting file (default is 24)
  -n OUTPUTFILE, --nameoftargetfile=OUTPUTFILE
                        name of the output file
  -w, --write           to write every single image to file
  -r, --reverse         iterate the files reversed
  -q, --quiet           the output should be hidden
  -m, --multiplerender  render the images multiple times
```

the `-w` will create a folder with all the images and then make a gif out of it with [ImageMagick](http://www.imagemagick.org/script/index.php)