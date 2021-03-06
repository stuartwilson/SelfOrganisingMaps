Note by Stuart Wilson - these images were taken from the topographica project, and the text from the original README file that accompanied them is below:




-----------------------------------------------------------------------------
The images in this directory were obtained from Harel Shouval and
Brian Blais on 16 April 1999; see the email below.  They were
converted from the cmp file format using plotcmp.m, and then saved as
PNG.  The two databases had some overlap, and they have been merged
using the following names:

base01.cmp       -> combined14.png
base02.cmp       -> combined15.png
base03.cmp       -> combined16.png
base04.cmp       -> combined17.png
base05.cmp       -> combined18.png
base06.cmp       -> combined01.png   
base07.cmp       -> combined19.png
base08.cmp       -> combined20.png
base09.cmp       -> combined21.png
base10.cmp       -> combined22.png
base11.cmp       -> combined02.png
base12.cmp       -> combined23.png
base13.cmp       -> combined03.png
base14.cmp       -> combined04.png
base15.cmp       -> combined05.png
base16.cmp       -> combined06.png
base17.cmp       -> combined07.png
base18.cmp       -> combined08.png
base19.cmp       -> combined09.png
base20.cmp       -> combined10.png
base21.cmp       -> combined11.png
base22.cmp       -> combined12.png
base23.cmp       -> combined24.png
base24.cmp       -> combined13.png

new_images01.cmp -> combined14.png
new_images02.cmp -> combined15.png
new_images03.cmp -> combined16.png
new_images04.cmp -> combined17.png
new_images05.cmp -> combined18.png
new_images06.cmp -> combined19.png
new_images07.cmp -> combined20.png
new_images08.cmp -> combined21.png
new_images09.cmp -> combined22.png
new_images10.cmp -> combined23.png
new_images11.cmp -> combined24.png
new_images12.cmp -> combined25.png

The combined dataset thus contains all images from both sets, with no
duplicates.

James A. Bednar
17 April 1999

-------------------------------------------------------------------------------
Date: Fri, 16 Apr 1999 10:29:44 -0400 (EDT)
From: Brian Blais <bblais@cns.brown.edu>
To: jbednar@cs.utexas.edu
CC: hzs@cns.brown.edu
In-reply-to: <Pine.GSO.3.96.990415173727.9146B-100000@cns.brown.edu>
        (hzs@cns.brown.edu)
Subject: Re: Image database? (fwd)

Hello,

        Your email to Harel requesting our image database was forwarded to
me.  It is no problem to get the database: we have placed the images on our
ftp server.  You can get them at: ftp://cns.brown.edu/pub/bblais/pics/

The original pictures used in the "Effect of Cortical Input Misalignment on
Ocular Dominance " (Neural Computation, volume 8) are in base24.cmp (or
base24_dog.cmp for the retinally preprocessed images).  The images used in
some of the more recent publications are in new_images12.cmp (and
new_images12_dog.cmp).

This .cmp format is a home-grown format which stores a series of images using
8 bit depth, and contains minimum and maximum values so that the 0 to 255 byte
values can get scaled to a different range.  There are some matlab utilities
in ftp://cns.brown.edu/pub/bblais/pics/utils for reading and writing this
format.  A full description of the format is in the cmpread.m file.

These databases are pretty limited, if you are trying to do some serious
statistics, but they work quite well for practically everything else.  For a
much (much) larger database, you can go to the Van Hateren Datapage at
http://hlab.phys.rug.nl/archive.html.   He has about 2000 images there, plus
some movies, and other things.  

If you have any questions or problems, please feel free to contact me.


                                      Brian Blais

