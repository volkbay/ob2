# ob2
GAN based generation from brushstrokes of an human artist. Mentioned artist is [Özgür Balli, PhD](https://www.instagram.com/ozgurballii/) from Hacettepe University Fine Arts. The software consist a basic work that learns many brushstroke samples of the artist to generate its own (Fig.1). The resulting images are representative composition (Fig.3-4). The core algorithm is an artist-critic type generative adversarial neural network (GAN).

>**Paper** This work is mentioned in [_Ballı, Ö. (2020). TRANSHÜMANİZM BAĞLAMINDA BİR YAPAY ZEKÂ SANATÇI UYGULAMASI: OBv2.Tykhe Sanat ve Tasarım Dergisi, 5 (9), 141-162._](https://dergipark.org.tr/tr/pub/tykhe/issue/58402/810977#article_cite)

![res47](https://user-images.githubusercontent.com/97564250/232256111-bd7cc032-141b-4cbc-ad5b-1d226735cde0.png)
![res48](https://user-images.githubusercontent.com/97564250/232256112-c0fdd73e-0009-48fd-a96a-e62d95bee20a.png)
![res49](https://user-images.githubusercontent.com/97564250/232256113-8c3cb8fb-5bf3-4fb7-976d-95a21e16c2d3.png)

_Fig. 1: Generated brushstrokes_

## :books: Dependencies
This is a Python3 projects using modules:
- OpenCV v3
- Tensorflow v2.1
- NumPy
- glob
- IPython
- Matplotlib

For GUI, PySide2 and win32api are needed.

## :computer: GUI
The interface is for simply composing generated brushstrokes onto a canvas. At this step, GAN has already trained and custom strokes are obtained. The user may select base strokes or let the software select them randomly.

<p>
  <img src="https://user-images.githubusercontent.com/97564250/232256153-e341eb46-ed62-4879-ac13-00ace76b92a0.jpg" width="49%">
</p>

_Fig. 2: User interface_

<p>
  <img src="https://user-images.githubusercontent.com/97564250/232256352-c8aa427d-8baa-4e78-9644-2d6c99efb967.jpg" width="49%">
  <img src="https://user-images.githubusercontent.com/97564250/232256354-81887395-91dd-45d1-83a7-a1b5d3be8568.jpg" width="49%">
</p>

_Fig. 3-4: Resulting composition_
