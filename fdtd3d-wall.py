#! coding:utf-8
"""
fdtd3d-wall.py
参考：日本音響学会　サイエンスシリーズ14「FDTD法で視る音の世界」　付録DVD

"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches


#データ保存用フォルダを作成
os.makedirs('data', exist_ok=True)
#画像保存用フォルダを作成
os.makedirs('fig', exist_ok=True)
files = [] #動画作成のための画像のファイル名保存用

# 変数の宣言
xmax = 4.000e0 # x軸解析領域 [m]
ymax = 5.000e0 # y軸解析領域 [m]
zmax = 3.000e0 # z軸解析領域 [m]
tmax = 2.000e-2# 解析時間 [s]
c0 = 3.435e2 # 空気の音速 [m/s]
dh = 1.500e-2 # 空間離散化幅 [m] 対称周波数の波長の1/10~1/20程度目安
dt = dh/(np.sqrt(3)*c0) # 時間離散化幅 [s] 安定条件dt <= dh/(root(n)*c0) n:次元数
row0 = 1.205e0 # 空気の密度 [kg/m^3]
##############################################################
#スピーカ位置（収録マイク位置と同じ）
xdr1 = 2.000e0 # x軸音源位置 [m]
ydr1 = 3.730e0  # y軸音源位置 [m]
zdr1 = 0.400e0 # y軸音源位置 [m]
##############################################################
M = 1 #マイク個数
mx1 = 0.2500e0 # x軸マイク位置 [m]
my1 = 3.730e0 # y軸マイク位置 [m]
mz1 = 0.450e0 #ｚ軸マイク位置 [m]
##############################################################
#障害物の諸条件
xon = 1.100e0 # 直方体x座標最小値 [m]
xox = 2.900e0 # 直方体x座標最大値 [m]
yon = 1.400e0 # 直方体y座標最小値 [m]
yox = 1.600e0 # 直方体y座標最大値 [m]
zon = 0.000e0 # 直方体z座標最小値 [m]
zox = 1.200e0 # 直方体z座標最大値 [m]
alpn = 0.000e0 # 直方体表面吸音率 [-]
m = 1.000e0 # ガウシアンパルス最大値 [m^3/s]
a = 2.000e6 # ガウシアンパルス係数 [-]
t0 = 2.000e-3 # ガウシアンパルス中心時間 [s]
pl = 16 # PML層数 [-]
pm = 4 # PML減衰係数テーパー乗数 [-]
emax = 1.200e0 # PML減衰係数最大値
fn = "out_3d" # 出力ファイルネーム
df = 10 # 出力ファイルスキップ数

# Buffer
ex = np.zeros(pl + 1)
pmla = np.zeros((pl + 1,1))
pmlb = np.zeros((pl + 1,1))
pmlc = np.zeros((pl + 1,1))

# 解析範囲
ix = int(xmax / dh) + pl * 2
jx = int(ymax / dh) + pl * 2
kx = int(zmax / dh) + pl * 2
tx = int(tmax / dt)

# 直方体位置(pml用のマージンも含まれている)
ion = int(xon / dh) + pl
iox = int(xox / dh) + pl
jon = int(yon / dh) + pl
jox = int(yox / dh) + pl
kon = int(zon / dh) + pl
kox = int(zox / dh) + pl
##############################################################
# 加振点位置
idr1 = int(xdr1 / dh) + pl
jdr1 = int(ydr1 / dh) + pl
kdr1 = int(zdr1 / dh) + pl
##############################################################
# マイク点位置
im1 = int(mx1 / dh) + pl
jm1 = int(my1 / dh) + pl
km1 = int(mz1 / dh) + pl
##############################################################

# 加振時間
#tdr = int((2.0 * t0) / dt)
tdr = int(tmax / dt)

# 体積弾性率
kp0 = row0 * c0 * c0

# 特性インピーダンス
z0 = row0 * c0

# 表面インピーダンス
if alpn != 0.0:
    zn = (row0 * c0 * (1.0 + np.sqrt(1.0 - alpn)) / 
    (1.0 - np.sqrt(1.0 - alpn)))

# Courant数
clf = c0 * dt / dh

# 粒子速度用更新係数
vc = clf / z0

# 音圧用更新係数
pc = clf * z0

# PML用更新係数
for i in range(pl):
    i += 1
    ex[i] = emax * np.power(float(pl - i + 1) / float(pl), float(pm))

pmla[1:pl,0] = (1.0 - ex[1:pl]) / (1.0 + ex[1:pl])
pmlb[1:pl,0] = clf / z0 / (1.0 + ex[1:pl])
pmlc[1:pl,0] = clf * z0 / (1.0 + ex[1:pl])

#行列演算用pml係数
maxIndex = max([ix,jx,kx])
pmlaBox = np.tile(pmla,(maxIndex+1,1,maxIndex+1))
pmlaBox = pmlaBox.transpose(1,0,2)
pmlbBox = np.tile(pmlb,(maxIndex+1,1,maxIndex+1))
pmlbBox = pmlbBox.transpose(1,0,2)
pmlcBox = np.tile(pmlc,(maxIndex+1,1,maxIndex+1))
pmlcBox = pmlcBox.transpose(1,0,2)

# メモリ格納
p = np.zeros((ix + 1, jx + 1, kx + 1))
px = np.zeros((ix + 1, jx + 1, kx + 1))
py = np.zeros((ix + 1, jx + 1, kx + 1))
pz = np.zeros((ix + 1, jx + 1, kx + 1))

vx = np.zeros((ix + 1, jx + 1, kx + 1))
vy = np.zeros((ix + 1, jx + 1, kx + 1))
vz = np.zeros((ix + 1, jx + 1, kx + 1))

q = np.zeros((tdr, 1)) #加振信号
p_m = np.zeros((tdr, 1)) #マイク収録信号

# 音源波形の生成
for t in range(tdr):
    # t={1,2,...,tdr}
    q[t,0] = m * np.exp(-a * pow(float(t * dt - t0), 2.0))

np.savetxt('qSingnal.csv',q,delimiter=',')

# 時間ループ
tcount = 1
fcount = 0
txstep = float(tx) / 100.

print("Time Loop Start" + os.linesep)

for t in range(tx):
    # t = {1,2,...,tx}
    
    print(t, tx-1)

    # ----------------------------
    # 粒子速度(vx)の更新
    # ----------------------------

    # PML
    vx[1:pl+1, 1:jx, 1:kx] = (pmlaBox[0:pl,1:jx,1:kx] * vx[1:pl+1, 1:jx, 1:kx] - 
        pmlbBox[0:pl,1:jx,1:kx] * (p[2:pl+2, 1:jx, 1:kx] - p[1:pl+1, 1:jx, 1:kx]))

    # 音響領域
    vx[pl+1:ix-pl, 1:jx, 1:kx] = (vx[pl+1:ix-pl, 1:jx, 1:kx] - vc * 
        (p[pl+2:ix-pl + 1, 1:jx, 1:kx] - p[pl+1:ix-pl, 1:jx, 1:kx]))

    # PML
    vx[ix-pl:ix, 1:jx, 1:kx] = (np.flip(pmlaBox[0:pl,1:jx,1:kx],0) * 
        vx[ix-pl:ix, 1:jx,1:kx] - np.flip(pmlbBox[0:pl,1:jx,1:kx],0) * 
        (p[ix-pl+1:ix + 1, 1:jx,1:kx] - p[ix-pl:ix, 1:jx,1:kx]))
    # ----------------------------
    # 粒子速度(vy)の更新
    # ----------------------------
    # PML
    vy[1:ix, 1:pl+1, 1:kx] = (pmlaBox[0:pl,1:ix,1:kx].transpose(1,0,2) * 
        vy[1:ix, 1:pl+1, 1:kx] - pmlbBox[0:pl,1:ix,1:kx].transpose(1,0,2) * 
        (p[1:ix, 2:pl+2, 1:kx] - p[1:ix, 1:pl+1, 1:kx]))
    #音響領域
    vy[1:ix, pl+1:jx-pl, 1:kx] = (vy[1:ix, pl+1:jx-pl, 1:kx] - vc * 
        (p[1:ix, pl+2:jx-pl + 1, 1:kx] - p[1:ix, pl+1:jx-pl, 1:kx]))
    #PML
    vy[1:ix,jx-pl:jx,1:kx] = (np.flip(pmlaBox[0:pl,1:ix,1:kx],0).transpose(1,0,2) * 
        vy[1:ix, jx-pl:jx,1:kx] - np.flip(pmlbBox[0:pl,1:ix,1:kx],0).transpose(1,0,2) * 
        (p[1:ix, jx-pl+1:jx + 1,1:kx] - p[1:ix, jx-pl:jx,1:kx]))
    # ----------------------------
    # 粒子速度(vz)の更新
    # ----------------------------
    # PML
    vz[1:ix+1,1:jx+1,1:pl+1] = (pmlaBox[0:pl,1:jx+1,1:ix+1].transpose(2,1,0) * 
        vz[1:ix+1, 1:jx+1, 1:pl+1] - pmlbBox[0:pl,1:jx+1,1:ix+1].transpose(2,1,0) * 
        (p[1:ix+1,1:jx+1, 2:pl+2] - p[1:ix+1,1:jx+1,1:pl+1]))
    #音響領域
    vz[1:ix+1,1:jx+1,pl+1:kx-pl] = (vz[1:ix+1,1:jx+1, pl+1:kx-pl] - vc * 
        (p[1:ix+1,1:jx+1, pl+2:kx-pl+1] - p[1:ix+1,1:jx+1,pl+1:kx-pl]))
    # PML
    vz[1:ix+1,1:jx+1,kx-pl:kx] = (np.flip(pmlaBox[0:pl,1:jx+1,1:ix+1],0).transpose(2,1,0) * 
        vz[1:ix+1,1:jx+1,kx-pl:kx] - np.flip(pmlbBox[0:pl,1:jx+1,1:ix+1],0).transpose(2,1,0) * 
        (p[1:ix+1, 1:jx+1,kx-pl+1:kx+1] - p[1:ix+1, 1:jx+1,kx-pl:kx]))

    # 境界条件(vx)の計算
    vx[0,1:jx] = 0.0
    vx[ix,1:jx] = 0.0

    if alpn != 0.0: #吸音率が０でない場合(完全反射でない場合)
        vx[ion - 1, jon:jox, kon:kox] = p[ion - 1, jon:jox, kon:kox] / zn
        vx[iox, jon:jox, kon:kox] = -p[iox + 1,jon:jox, kon:kox] / zn

    else: #完全反射(剛壁)のとき、音圧最大、粒子速度最小=0
        vx[ion - 1, jon:jox, kon:kox] = 0.0
        vx[iox, jon:jox, kon:kox] = 0.0

    # 境界条件(vy)の計算
    vy[1:ix, 0, 1:kx] = 0.0
    vy[1:ix, jx, 1:kx] = 0.0

    if alpn != 0.0:
        vy[ion:iox, jon - 1, kon:kox] = p[ion:iox, jon - 1, kon:kox] / zn
        vy[ion:iox, jox, kon:kox] = -p[ion:iox, jox + 1, kon:kox] /zn
    else:
        vy[ion:iox, jon - 1, kon:kox] = 0.0
        vy[ion:iox, jox, kon:kox] = 0.0

    # 境界条件(vz)の計算   
    vz[1:ix + 1, 1:jx + 1, 0] = 0.0
    vz[1:ix + 1, 1:jx + 1, kx] = 0.0

    if alpn != 0.0:
        vz[ion:iox+1, jon:jox+1, kon - 1] = p[ion:iox + 1, jon:jox+1, kon - 1] / zn
        vz[ion:iox+1, jon:jox+1, kox] = -p[ion:iox+1, jon:jox+1, kox + 1] / zn
    else:
        vz[ion:iox+1, jon:jox+1, kon - 1] = 0.0
        vz[ion:iox+1, jon:jox+1, kox] = 0.0

##################################################################################################################
    # ----------------------------
    # 音圧(px)の更新
    # ----------------------------
    px[1:pl+1, 1:jx,1:kx] = (pmlaBox[0:pl,1:jx,1:kx]  * px[1:pl+1, 1:jx,1:kx] - pmlcBox[0:pl,1:jx,1:kx] * 
        (vx[1:pl+1, 1:jx,1:kx] - vx[0:pl, 1:jx,1:kx]))
    px[pl+1:ix-pl+1, 1:jx, 1:kx] = px[pl+1:ix-pl+1, 1:jx, 1:kx] - pc * (vx[pl+1:ix-pl+1, 1:jx, 1:kx] - vx[pl:ix-pl, 1:jx, 1:kx])

    #音響加振
    if t < tdr:
        px[idr1, jdr1, kdr1] = px[idr1, jdr1, kdr1] + dt * kp0 * q[t,0] / 3.0 / (dh * dh * dh)
               
    px[ix-pl+1:ix+1, 1:jx, 1:kx] = (np.flip(pmlaBox[0:pl,1:jx,1:kx],0) * px[ix-pl+1:ix+1, 1:jx, 1:kx] - 
        np.flip(pmlcBox[0:pl,1:jx,1:kx],0) * (vx[ix-pl+1:ix+1, 1:jx, 1:kx] - vx[ix-pl:ix, 1:jx, 1:kx]))

    # ----------------------------
    # 音圧(py)の更新    
    # ----------------------------
    py[1:ix, 1:pl+1, 1:kx] = (pmlaBox[0:pl,1:ix,1:kx].transpose(1,0,2) * py[1:ix, 1:pl+1, 1:kx] - 
        pmlcBox[0:pl,1:ix,1:kx].transpose(1,0,2) * (vy[1:ix, 1:pl+1, 1:kx] - vy[1:ix, 0:pl, 1:kx]))
    py[1:ix,pl+1:jx-pl+1, 1:kx] = py[1:ix, pl+1:jx-pl+1, 1:kx] - pc * (vy[1:ix, pl+1:jx-pl+1, 1:kx] - vy[1:ix, pl:jx-pl, 1:kx])
    #加振点#########################################################################################
    if t < tdr:
        py[idr1,jdr1,kdr1] = py[idr1,jdr1,kdr1] + dt * kp0 * q[t,0] / 3.0 / (dh * dh * dh)
        
    py[1:ix,jx-pl+1:jx+1,1:kx] = (np.flip(pmlaBox[0:pl,1:ix,1:kx],0).transpose(1,0,2) * py[1:ix, jx-pl+1:jx+1,1:kx] - 
        np.flip(pmlcBox[0:pl,1:ix,1:kx],0).transpose(1,0,2) * (vy[1:ix, jx-pl+1:jx+1,1:kx] - vy[1:ix, jx-pl:jx,1:kx]))

    # ----------------------------
    # 音圧(pz)の更新
    # ----------------------------
    pz[1:ix+1,1:jx+1,1:pl+1] = (pmlaBox[0:pl,1:jx+1,1:ix+1].transpose(2,1,0) * pz[1:ix+1,1:jx+1,1:pl+1] - 
        pmlcBox[0:pl,1:jx+1,1:ix+1].transpose(2,1,0) * (vz[1:ix+1,1:jx+1,1:pl+1] - vz[1:ix+1,1:jx+1,0:pl]))
    pz[1:ix+1,1:jx+1,pl+1:kx-pl+1] = pz[1:ix+1,1:jx+1,pl+1:kx-pl+1] - pc * (vz[1:ix+1,1:jx+1, pl+1:kx-pl+1] - vz[1:ix+1,1:jx+1,pl:kx-pl])
    #加振点#########################################################################################
    if t < tdr:
        pz[idr1,jdr1,kdr1] = pz[idr1,jdr1,kdr1] + dt * kp0 * q[t,0] / 3.0 / (dh * dh* dh)
          
    pz[1:ix+1,1:jx+1,kx-pl+1:kx+1] = (np.flip(pmlaBox[0:pl,1:jx+1,1:ix+1],0).transpose(2,1,0) * pz[1:ix+1,1:jx+1,kx-pl+1:kx+1] - 
        np.flip(pmlcBox[0:pl,1:jx+1,1:ix+1],0).transpose(2,1,0) * (vz[1:ix+1,1:jx+1,kx-pl+1:kx+1] - vz[1:ix+1,1:jx+1,kx-pl:kx]))

    # 音圧の合成
    p = px + py + pz
    # マイク位置音圧
    p_m[t,0] = p[im1,jm1,km1] 
    ###########################################################################
    # df回ごとにデータ(p)と図を書き出す
    if t % df == 0:
        maxamp = np.max(np.abs(q))* dt * kp0 *100#駆動信号の最大値
        data = p[:,:,kdr1] #平面プロット
        x_ticks = np.arange(-pl*dh,data.shape[0]*dh-pl*dh,dh)
        y_ticks = np.arange(-pl*dh,data.shape[1]*dh-pl*dh,dh)
        X,Y = np.meshgrid(x_ticks, y_ticks)
        X = X/100 #cm->m
        Y = Y/100 #cm->m
        fig = plt.figure(figsize=(9,16))
        ax = fig.add_subplot(211, aspect='equal')
        sc = ax.imshow(data, vmin=-maxamp, vmax=maxamp, extent=[-pl*dh, ymax+pl*dh, -pl*dh, xmax+pl*dh], origin='lower')
        fig.colorbar(sc)
        
        patch_xy = patches.Rectangle(xy=(yon, xon), width=yox-yon, height=xox-xon, angle=0,facecolor='b',edgecolor='b')
        ax.add_patch(patch_xy)
        patch_pml_xy = patches.Rectangle(xy=(0, 0), width=ymax, height=xmax, angle=0,facecolor=[0,0,0,0],edgecolor='red',linewidth=2,linestyle=':')
        ax.add_patch(patch_pml_xy)
        
        plt.title('X-Y cross surface')
        plt.xlabel('Y [m]')
        plt.ylabel('X [m]')
        
        data = p[idr1,:,:].T #断面プロット
        ax = fig.add_subplot(212, aspect='equal')
        sc = ax.imshow(data,vmax=maxamp,vmin=-maxamp,extent=[-pl*dh, ymax+pl*dh, -pl*dh, zmax+pl*dh], origin='lower')
        fig.colorbar(sc)
        patch_yz = patches.Rectangle(xy=(yon, zon), width=yox-yon, height=zox-zon, angle=0,facecolor='b',edgecolor='b')
        ax.add_patch(patch_yz)
        patch_pml_xy = patches.Rectangle(xy=(0, 0), width=ymax, height=zmax, angle=0,facecolor=[0,0,0,0],edgecolor='red',linewidth=2,linestyle=':')
        ax.add_patch(patch_pml_xy)
        
        plt.tight_layout()
        plt.title('Y-Z cross surface')
        plt.xlabel('Y [m]')
        plt.ylabel('Z [m]')
        
        # save as png
        plt.savefig('fig/figure'+'{0:03d}'.format(t)+'.png')
        
        files.append('figure'+'{0:03d}'.format(t)+'.png')
        print('Data was exported')

    if (t >= int(float(tcount) * txstep)):
        print("%s%7.2f%s%s" % ("Completed", float(t) / float(tx) * 100., "%", os.linesep))
        tcount += 1
        
#ポスト処理
#マイク位置音圧
plt.figure()
plot_t = np.arange(0,tmax-dt,dt)
plt.plot(plot_t,np.arange(tdr), p_m[:,0])
plt.show()

    
    


