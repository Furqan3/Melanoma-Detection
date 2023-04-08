import numpy as np
import cv2 as cv 
import pandas as pd
import matplotlib.pyplot as plt



def asymmetry(image):
    obj = np.where(image != 0)
    center_x = int(np.mean(obj[0]))
    center_y = int(np.mean(obj[1]))
    x_min = np.min(obj[0])
    x_max = np.max(obj[0])
    y_min = np.min(obj[1])
    y_max = np.max(obj[1])
    
    x_left_counts = [np.count_nonzero(image[i, :]) for i in range(x_min, center_x)]
    x_right_counts = [np.count_nonzero(image[i, :]) for i in range(center_x, x_max)]
    y_left_counts = [np.count_nonzero(image[:, i]) for i in range(y_min, center_y)]
    y_right_counts = [np.count_nonzero(image[:, i]) for i in range(center_y, y_max)]
    
    x_left_counts = np.array(x_left_counts)
    x_right_counts = np.array(x_right_counts)
    y_left_counts = np.array(y_left_counts)
    y_right_counts = np.array(y_right_counts)
    x_left_counts = x_left_counts[x_left_counts != 0]
    x_right_counts = x_right_counts[x_right_counts != 0]
    y_left_counts = y_left_counts[y_left_counts != 0]
    y_right_counts = y_right_counts[y_right_counts != 0]
    x_left_probs = x_left_counts / np.sum(x_left_counts)
    x_right_probs = x_right_counts / np.sum(x_right_counts)
    y_left_probs = y_left_counts / np.sum(y_left_counts)
    y_right_probs = y_right_counts / np.sum(y_right_counts)
    
    x_left_entropy = np.sum(x_left_probs * np.log(x_left_probs))
    x_right_entropy = np.sum(x_right_probs * np.log(x_right_probs))
    y_left_entropy = np.sum(y_left_probs * np.log(y_left_probs))
    y_right_entropy = np.sum(y_right_probs * np.log(y_right_probs))
    
    asymmetry = (x_left_entropy + x_right_entropy + y_left_entropy + y_right_entropy) / 4
    
    return asymmetry


def border(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv.erode(gray, kernel, iterations=1)
    diff = cv.subtract(gray, eroded)
    border_length = np.count_nonzero(diff == 255)
    area = cv.countNonZero(gray)
    circularity = 4 * np.pi * area / (border_length ** 2)
    return circularity

def color(image,mask):
    image=apply_mask(image,mask)
    image=cv.cvtColor(image,cv.COLOR_BGR2YUV)
    y,u,v=cv.split(image)
    y_mean=np.mean(y)
    u_mean=np.mean(u)
    v_mean=np.mean(v)
    return np.std([y_mean,u_mean,v_mean])

def apply_mask(image, mask):
    image = cv.bitwise_and(image, mask)
    return image

def dimeter(mask):
    mask=cv.cvtColor(mask,cv.COLOR_BGR2GRAY)
    pixels=np.where(mask==255)
    x=np.max(pixels[0])-np.min(pixels[0])
    y=np.max(pixels[1])-np.min(pixels[1])
    dimeter=np.sqrt(x**2+y**2)
    return dimeter

def plot_diagrams(asm,bor,col,dim):
    # Boxplot of the features
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].boxplot([asm[:161], asm[161:201]], labels=('Normal', 'Melanoma'))

    axs[0, 0].set_title('Asymmetry')

    axs[0, 1].boxplot([bor[:161], bor[161:201]], labels=('Normal', 'Melanoma'))
    axs[0, 1].set_title('Border')

    axs[1, 0].boxplot([col[:161], col[161:201]], labels=('Normal', 'Melanoma'))
    axs[1, 0].set_title('Color')

    axs[1, 1].boxplot([dim[:161], dim[161:201]], labels=('Normal', 'Melanoma'))
    axs[1, 1].set_title('Diameter')

#histogram of the features
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].hist([asm[:161],  asm[161:201]], label=('Normal',  'Melanoma'))
    
    axs[0, 0].set_title('Asymmetry')
    
    axs[0, 1].hist([bor[:161],  bor[161:201]], label=('Normal', 'Melanoma'))
    axs[0, 1].set_title('Border')
    
    axs[1, 0].hist([col[:161],  col[161:201]], label=('Normal',  'Melanoma'))
    axs[1, 0].set_title('Color')
    
    axs[1, 1].hist([dim[:161] , dim[161:201]], label=('Normal',  'Melanoma'))
    axs[1, 1].set_title('Diameter')
    
    plt.legend()
    
#scatter plot of the features

    y = range(200)
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].scatter(range(len(asm[:161])), asm[:161], label='Normal')
    axs[0, 0].scatter(range(len(asm[161:201])), asm[161:201], label='Melanoma')
    axs[0, 0].legend()
    axs[0, 0].set_title('Asymmetry')

    axs[0, 1].scatter(range(len(bor[:161])), bor[:161], label='Normal')
    axs[0, 1].scatter(range(len(bor[161:201])), bor[161:201], label='Melanoma')
    axs[0, 1].legend()
    axs[0, 1].set_title('Border')

    axs[1, 0].scatter(range(len(col[:161])), col[:161], label='Normal')
    axs[1, 0].scatter(range(len(col[161:201])), col[161:201], label='Melanoma')
    axs[1, 0].legend()
    axs[1, 0].set_title('Color')

    axs[1, 1].scatter(range(len(dim[:161])), dim[:161], label='Normal')
    axs[1, 1].scatter(range(len(dim[161:201])), dim[161:201], label='Melanoma')
    axs[1, 1].legend()
    axs[1, 1].set_title('Diameter')

    plt.show()    

def print_data(asm,bor,col,dim):
    data = {'Feature': ['Asymmetry', 'Asymmetry', 'Border', 'Border', 'Color', 'Color', 'Diameter', 'Diameter'],
            'Subgroup': ['Normal', 'Melanoma', 'Normal', 'Melanoma', 'Normal', 'Melanoma', 'Normal', 'Melanoma'],
            'Mean + Std': [np.mean(asm[:161])+np.std(asm[:161]), np.mean(asm[161:201])+np.std(asm[161:201]), 
                            np.mean(bor[:161])+np.std(bor[:161]), np.mean(bor[161:201])+np.std(bor[161:201]),
                            np.mean(col[:161])+np.std(col[:161]), np.mean(col[161:201])+np.std(col[161:201]), 
                            np.mean(dim[:161])+np.std(dim[:161]), np.mean(dim[161:201])+np.std(dim[161:201])],
            'Mean - Std': [np.mean(asm[:161])-np.std(asm[:161]), np.mean(asm[161:201])-np.std(asm[161:201]), 
                            np.mean(bor[:161])-np.std(bor[:161]), np.mean(bor[161:201])-np.std(bor[161:201]),
                            np.mean(col[:161])-np.std(col[:161]), np.mean(col[161:201])-np.std(col[161:201]), 
                            np.mean(dim[:161])-np.std(dim[:161]), np.mean(dim[161:201])-np.std(dim[161:201])]
            }

    df = pd.DataFrame(data)
    print(df)

def mainloop():
    xl=pd.read_excel('PH2Dataset/PH2_Dataset.xlsx')
    names=xl.iloc[:,0]
    asm,bor,col,dim=[],[],[],[]
    r=0
    wr=0
    for i in names:
        image = cv.imread(f'PH2Dataset/PH2_Dataset_images/{i}/{i}_Dermoscopic_Image/{i}.bmp')
        mask = cv.imread(f'PH2Dataset/PH2_Dataset_images/{i}/{i}_lesion/{i}_lesion.bmp')
        a=asymmetry(apply_mask(image,mask))
        b=border(mask)
        c=color(image,mask)
        d=dimeter(mask)
        asm.append(a)
        bor.append(b)
        col.append(c)
        dim.append(d)
        row,_=np.where(xl==i)
        if a<=-5.388027 and b>=-1.314408 and c<=26.099766 and d>=691.211341 :
            if(xl.iloc[row,3] == 'X').any():
                r+=1
            else:
                wr+=1
        elif (xl.iloc[row,3] == 'X').any():
            wr+=1
        else:
            r+=1
    print('corrected predictions: ',r)
    print('wrong predictions: ',wr)
    print('Accuracy',r/(r+wr)*100,'%')
    print_data(asm,bor,col,dim)
    plot_diagrams(asm,bor,col,dim)



if __name__=='__main__':
    mainloop()
