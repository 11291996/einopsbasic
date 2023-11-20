import torch
from einops import rearrange, reduce, repeat
#rearranging tensors(numpy, torch tensor, tensorflow)
a = torch.randint(0, 255, (4, 64, 64, 3))
#a[i,j,k] -> a[j,i,k]
rearrange(a[0], 'i j k -> j i k') #unlike torch.einsum index must be sperated
rearrange(a[0], 'i j ... k -> i j k (...)') #the same as adding () only
#concatenating a batch following y axis
#make new[b * i, j, k] -> 
#n = 0 
#for m in range(b): 
#   for p in range(i): 
#       a[m, i, j ,k] = new[p + n, j, k]
#   n+=i
#concatenating a batch following x axis 
rearrange(a, 'b i j k -> j (b j) k')
#flattening all elements
rearrange(a, 'b i j k -> (b i j k)')
#resizing the batch #(b,) -> (b1, b2)
#new[b1, b2, i, j, k]
#for s1 in range(b1):
#   for s2 in range(b2):
#       a[s1 + s2, i, j, k] = new[s1, s2, i, j, k]
rearrange(a, '(b1, b2) i j k -> b1, b2, i j k', b1 = 2)
#concatenate batches
rearrange(a, '(b1, b2) i j k -> (b1, i), (b2, j) k', b1 = 2)
#concatenate on other dimensions
rearrange(a, 'b h (w w2) c -> (h w2) (b w) c', w2=2)
#dimensions in () is important #variable in for loop changes
rearrange(a, 'b h w c -> h (w b) c') #gives a different result from above
#the order of a[i] is different
rearrange(a, '(b1 b2) h w c -> h (b1 b2 w) c', b1=2)
rearrange(a, '(b1 b2) h w c -> h (b2 b1 w) c', b1=2)
#reduce a dimension by average each element 
reduce(a, 'b h w c -> h w c', 'mean') #min, max, prod, sum are also possible
reduce(a, 'b (h h2) (w w2) c -> h (b w) c', 'max', h2=2, w2=2) #max pooling application
reduce(a, 'b h w c -> b () () c', 'max') #() means retaining the dimension equal with 1 
#copying elements
repeat(a[0], 'h w c -> h new_axis w c', new_axis=5) #right side of new axis is copied 
#along dimension is possible
repeat(a[0], 'h w c -> (2 h) (2 w) c')
