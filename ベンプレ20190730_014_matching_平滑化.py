# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 19:57:23 2019

@author: Takahiro
"""

import sys
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

def main():
    mass = 50 #プレートの重さ(kg)
    real_height = 1.9 #画面の縦方向の実寸（m)

    #threshold = 0.8 #画像マッチングの閾値(最初0.68)
    rSize = 1 #白塗りのサイズ（templateの何倍？)
    cnt = 0      # カウント変数
    n_diff = 4  #差分の間隔（大きいほど平滑化）
    
    # カメラのキャプチャ
    fname = "benpre50.mp4"
    cap = cv2.VideoCapture(fname)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 総フレーム数
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_ps = int(cap.get(cv2.CAP_PROP_FPS))
    
    meters_per_pix = real_height / frame_height    
    #この画像とマッチする箇所を元画像から探索
    template_org = cv2.imread('./matching/matching_jiku1.jpg', 0)
    cv2.imshow("template_org", template_org)
    template = template_org
    #template = cv2.cvtColor(template_org, cv2.COLOR_BGR2GRAY)
    temp_w, temp_h = template_org.shape[::-1]
    #座標格納用のデータフレーム作成
    zahyou_df = pd.DataFrame( columns=['frame','x','y'] )

    # Define the Codec and File Name
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('./results/result_' + fname, fourcc, 210, (640,360))
    #while(cap.isOpened()):
    for i in range(0, frame_count):
            # フレームの取得
            ret,frame = cap.read()
            # グレースケール変換
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #cv2.imwrite('./frames2/frame' + str(cnt) + '.jpg', gray)
            
            #templateとマッチする箇所を元画像から探索
            res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            pt = max_loc    
            cv2.rectangle(frame, pt, (pt[0] + int(temp_w * rSize), \
                                      pt[1] + int(temp_h * rSize)), \
                                      (255,255,255), -1)
            #座標格納            
            tmp_se = pd.Series( [ i, pt[0], pt[1] ], index=zahyou_df.columns )
            zahyou_df = zahyou_df.append( tmp_se, ignore_index=True )
            # Write the frame
            out.write(frame)
            cv2.imshow("Frame", frame)
            
            print('i=' + str(i) + ' / ' + str(frame_count-1))
            print('cnt=' + str(cnt))
            #k = cv2.waitKey(1) & 0xFF
            cnt += 1    # カウントを1増やす
                            
            # qキーが押されたら途中終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    cap.release()
    cv2.destroyAllWindows()
    zahyou_df['x'] = zahyou_df['x'].astype(int)
    zahyou_df['y'] = zahyou_df['y'].astype(int)
    
    ymax = zahyou_df['y'].max()  # y座標の最大値
    
    zahyou_df['t'] = zahyou_df['frame'] / frame_ps
    zahyou_df['h'] = meters_per_pix * (ymax - zahyou_df['y'])
    zahyou_df['v'] = frame_ps * zahyou_df['h'].diff(n_diff) / n_diff
    # 位置エネルギー
    zahyou_df['Ep'] = mass * 9.8* zahyou_df['h']
    # 運動エネルギー
    zahyou_df['Ek'] = mass * zahyou_df['v'] * zahyou_df['v'] / 2
    # 力学的エネルギー
    zahyou_df['Energy'] = zahyou_df['Ep'] + zahyou_df['Ek'] 
    #　仕事率
    zahyou_df['Power'] = zahyou_df['Energy'].diff(n_diff) / n_diff
    
    plt.figure()
    zahyou_df.plot( y=['h'], figsize=(16,4), color = ['#005b18'], legend=False, grid=True, alpha=0.5, title="Height(m)")
    plt.savefig('./results/' + fname + '_graph_h.png')
    zahyou_df.plot( y=['v'], figsize=(16,4), color = ['#9e151c'], legend=False, grid=True, alpha=0.5, title="Speed(m/s)")
    plt.savefig('./results/' + fname + '_graph_v.png')
    zahyou_df.plot( y=['Ep'], figsize=(16,4), color = ['#005b18'], legend=False, grid=True, alpha=0.5, title="Ep(J)")
    plt.savefig('./results/' + fname + '_graph_Ep.png')
    zahyou_df.plot( y=['Ek'], figsize=(16,4), color = ['#9e151c'], legend=False, grid=True, alpha=0.5, title="Ek(J)")
    plt.savefig('./results/' + fname + '_graph_Ek.png')
    zahyou_df.plot( y=['Energy'], figsize=(16,4), color = ['#FF0000'], legend=False, grid=True, alpha=0.5, title="Energy")
    plt.savefig('./results/' + fname + '_graph_Energy.png')
    zahyou_df.plot( y=['Power'], figsize=(16,4), color = ['#0000FF'], legend=False, grid=True, alpha=0.5, title="Power(W)")
    plt.savefig('./results/' + fname + '_graph_Power.png')

    zahyou_df.to_csv('./results/' + fname + '.csv')
    
if __name__ == '__main__':
    main()
