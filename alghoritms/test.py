from alghoritms import bubleSort, strFind_naive
a = [2,1,5,3,9,10,4,5,3,2,1,5,8,23,7,0,54,21,43,12,23,1,2,3,4,9,5,2,1,5,2,0,32,54,1,45,78,12,32,93,76,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,11,1,1,11,1,1,1,11,1,11,1,1,11,1,11,1,11,1,11,11,1]
b = [2,1,5,3,9,10,3,5,4]
c = bubleSort(b, reverse=True)
print(c)
pattern = "pa"
text = "parapapa"
text1 = "parapapaparapraspaarpapara"
print(strFind_naive(pattern, text))
print(strFind_naive(pattern, text1))