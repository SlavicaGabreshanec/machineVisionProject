<b>ПРОЕКТ ПО ПРЕДМЕТОТ МАШИНСКА ВИЗИЈА</b>

ТЕМА НА ПРОЕКТОТ: ДЕТЕКЦИЈА И СЕГМЕНТАЦИЈА НА РЕГИСТЕРСКИ ТАБЛИЧКИ НА СЛИКИ СО АВТОМОБИЛИ
<hr>
КРАТОК ОПИС НА АЛГОРИТАМОТ ЗА НАОЃАЊЕ НА РЕГИСТЕРСКИТЕ ТАБЛИЧКИ НА СЛИКИ СО АВТОМОБИЛИ:
<br></br>

 1: Почеток
 
 2: Влез: Оригинална слика
 
 3: Излез: Регистерска табличка
 
 4: Метод: K-Nearest Neighbors 
 
 5: Конвертирање RGB слика во Grayscale
 
 6: Филтер Morphological Transformation
 
 7: Трансформирање на Grayscale слика во бинарна слика
 
 8: Гаусов филтер за замаглување на слика
 
 9: Наоѓање на сите контури во сликата

10: Барање и распознавање на сите можни карактери во сликата

11: Отсекување на дел од сликата со највисок кандидат за да е регистерска табличка

12: Отсекување на регистерската табличка од оригиналната слика

<hr>
Главна цел на овој проект е да се направи едноставна детекција на регистерска табличка на возила и да се сегментира, односно да се извади слика само од регистрацијата.
Сите слики кои се наоѓаат во делот <b>LicencePlatePhotos</b> се слики од возила различно поставени, со различни регистерски таблички. За реализација на детекцијата и сегметирањето се користи <b>OpenCV</b>.

За детекција на карактери кои ги има во регистерските таблички е користено <b>K-Nearest Neighbors алгоритамот</b> кој се користи за класификација и регресија. 
Co методот <b>loadDataAndTrainKNN()</b> се врши вчитување на класификацијата и на сликите за тренирање во .txt формат и се тренира со помош на КNN. 
Оваа функција враќа True доколку е успешно тренирањето. Во <b>originalPhoto</b> ја имаме сликата за која што сакаме да сегментираме регистерска табличка.

За детекција на можни регистерски таблички на сликата се користи функцијата <b>detectPlatesInPhoto(originalPhoto)</b> која се наоѓа во <b>DetectLicPlates.py</b>. 
Секако најпрво мора сликата да ја направиме во grayscale, па потоа и thresholding на сликата.

Во функцијата <b>preprocess()</b> ја имаме постапката како се креира слика во grayscale и thresh.
Со помош на <b>HSV репрезентацијата на бои</b>, одност hue, saturation и value се добиваат сиви нијанси, од бела до црна.
За максимизирање на контрастите за повисока изразеност на рабовите е направено со помош на <b>cv2.morphologyEx()</b> од openCV.
Морфолошките трансформации се едноствни операции базирани на image shape, каде најчесто се применуваат на бинарни слики. 
Бараат два влеза, оригиналната сликата и вториот влез е структурен елемент кој укажува на природата на операцијата. 
И за крај се прави замаглување на сликата со помош на <b>Гаусовото замаглување</b> на слики.
И потоа се добива и threshold од замаглената слика со помош на <b>cv2.adaptiveThreshold()</b> од openCV.

Функцијата <b>findPossibleChars()</b> враќа можни карактери кои може да ги најде на сликата со направен threshold.
Се наоѓаат најпрво контурите со помош на функцијата <b>cv2.findContours()</b>.
Можен карактер се наоѓа со тоа што најпрво со помош на <b>cv2.boundingRect(countour)</b> се наоѓа одредени височини, широчини на контурата како и цела ареа со исцртување на правоаголник.
Во функцијата <b>checkIfPossibleChar()</b> кој се наоѓа во <b>DetectChars.py</b> се прави груб преглед на тоа дали контурата е char со веќе предефинираните MIN_PIXEL_AREA,  MIN_PIXEL_WIDTH, MIN_PIXEL_HEIGHT,
MIN_ASPECT_RATIO и MAX_ASPECT_RATIO, и оваа функција враќа true или false. 
Функцијата <b>findPossibleChars()</b> враќа листа од можни карактери. 

Во следната функција <b>listsOfMatchingChars()</b> главната цел е да се направи листа од листа со можни карактери кои се преклопуваат. 
Како влез во оваа функција е листата од можни карактери од претходната функција. Споредбата се прави во функцијата <b>findListOfMatchingChars(possibleChar, listOfChars)</b>. 
Во неа се прават пресметки за дистанцата помеѓу карактерите, аголот помеѓу нив и за крај се прави споредба со предефiнираните вредности за агол, за промена во висина, долзина, ширина итн. 

Во <b>getPlate()</b> се пресметува централната точка, височина и широчина на регистрацијата, точниот агол на регистрацијата и на крај се враќа листа од можни регистерски таблички.
И за крај на главната функција <b>detectPlatesInPhoto(originalPhoto)</b> во која беа сите овие фукциии повикани се исцртуваат линиите на правоаголникот околу можните регистерски таблички на сликата.

<b>Main функцијата</b> имаме уште едно повикување на функцијата <b>detectCharsInPlates(listOfPossiblePlates)</b> со помош на која се враќа конечна листа од можни регистерски таблички. 
За секоја можна табличка се прават следните проверки: најпрво со функцијата <b>preprocess()(објаснета погоре)</b> се зема grayscale и threshold на можната табличка. Следно се зголемува сликата со пронајден threshold за полесно наоѓање на карактерите.
Се прави уште еден threshold на сликата со веќе направен threshold за да се избегнат сиви делови.
Со повик на функцијата <b>findPossibleCharsInPlate()</b> се добиваат сите можни карактери во табличката, со првично наоѓање на контурите и ги враќа само контурите кои можат да бидат карактер.
И оваа функција враќа листа од можни таблички. 

Во главната функција имаме сортирање на можните таблички во опаѓачки редослед. 
И на крај со функцијата <b>drawAroundPlate(originalPhoto, licencePlate)</b> се исцртува црвен правоаголник околу регистерстата табличка. И крајниот резултат е претставување на оригиналната слика со обележана регистерска табличка и сегментирање само на регистерската табличка.
