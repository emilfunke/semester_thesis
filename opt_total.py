import pandas as pd
from tqdm import tqdm

header_list = ["img_path", "x", "y", "face_roi", "eyes_roi", "opt_path"]
total_df = pd.read_csv("/Users/PBL/PycharmProjects/sem_10/semester_thesis/full_face/total.csv", names=header_list)

print(total_df.shape)

len_eyes = 0
len_face = 0
with tqdm(total=total_df.shape[0]) as pbar:
    for index, row in total_df.iterrows():
        img_path, x, y, face_roi, opt_path = row['img_path'], row['x'], row['y'], row['face_roi'], row['opt_path']
        eyes_roi = row['eyes_roi']
        eyes_roi = eval(eyes_roi)
        face_roi = eval(face_roi)
        ex, ey = eyes_roi[0], eyes_roi[2]
        ew, eh = eyes_roi[1] + eyes_roi[3] - ex, eyes_roi[3] + eyes_roi[4] - ey
        fx, fy, fw, fh = face_roi[0], face_roi[1], face_roi[2], face_roi[3]
        eyes_roi = [ex, ey, ew, eh]

        opt_flow_df = pd.read_csv("/Users/PBL/PycharmProjects/sem_10/semester_thesis/" + opt_path)
        opt_flow_df = opt_flow_df.drop(columns=[' 1st best', ' 2nd best'])
        opt_flow_df = opt_flow_df.rename(columns={'# x': 'x', ' y': 'y', ' dx': 'dx', ' dy': 'dy'})

        opt_flow_df_eyes = opt_flow_df.drop(opt_flow_df[opt_flow_df.x < ex].index)
        opt_flow_df_eyes = opt_flow_df_eyes.merge(opt_flow_df_eyes.drop(
            opt_flow_df_eyes[opt_flow_df_eyes.y < ey].index))
        opt_flow_df_eyes = opt_flow_df_eyes.merge(
            opt_flow_df_eyes.drop(opt_flow_df_eyes[opt_flow_df_eyes.x > ex + ew].index))
        opt_flow_df_eyes = opt_flow_df_eyes.merge(
            opt_flow_df_eyes.drop(opt_flow_df_eyes[opt_flow_df_eyes.y > ex + eh].index))
        if len_eyes < len(opt_flow_df_eyes):
            len_eyes = len(opt_flow_df_eyes)

        opt_flow_df_face = opt_flow_df.drop(opt_flow_df[opt_flow_df.x < fx].index)
        opt_flow_df_face = opt_flow_df_face.merge(opt_flow_df_face.drop(
            opt_flow_df_face[opt_flow_df_face.y < fy].index))
        opt_flow_df_face = opt_flow_df_face.merge(
            opt_flow_df_face.drop(opt_flow_df_face[opt_flow_df_face.x > fx + fw].index))
        opt_flow_df_face = opt_flow_df_face.merge(
            opt_flow_df_face.drop(opt_flow_df_face[opt_flow_df_face.y > fy + fh].index))
        if len_face < len(opt_flow_df_face):
            len_face = len(opt_flow_df_face)

        opt_flow = [opt_flow_df_eyes, opt_flow_df_face]

        total_df.at[index, 'eyes_roi'] = eyes_roi
        total_df.at[index, 'opt_path'] = opt_flow

        pbar.set_postfix(len_eyes=len_eyes, len_face=len_face)
        pbar.update(1)

with tqdm(total=total_df.shape[0]) as pbar:
    for index, row in total_df.iterrows():
        opt_flow_df_eyes, opt_flow_df_face = row[5]

        eyes_shape = opt_flow_df_eyes.shape[1]
        face_shape = opt_flow_df_face.shape[1]
        current_eyes = opt_flow_df_eyes.shape[0]
        current_face = opt_flow_df_face.shape[0]

        changed_eyes = False
        zero_df = pd.DataFrame([[0]*eyes_shape], columns=opt_flow_df_eyes.columns)
        while current_eyes < len_eyes:
            opt_flow_df_eyes = opt_flow_df_eyes.append(zero_df, ignore_index=True)
            current_eyes = opt_flow_df_eyes.shape[0]
            changed_eyes = True

        changed_face = False
        zero_df = pd.DataFrame([[0]*face_shape], columns=opt_flow_df_face.columns)
        while current_face < len_face:
            opt_flow_df_face = opt_flow_df_face.append(zero_df, ignore_index=True)
            current_face = opt_flow_df_face.shape[0]
            changed_face = True

        opt_flow = [opt_flow_df_eyes.to_numpy(), opt_flow_df_face.to_numpy()]
        total_df.at[index, 'opt_path'] = opt_flow

        pbar.set_postfix(changed_eyes=changed_eyes, changed_face=changed_face)
        pbar.update(1)

print(total_df.shape)
total_df.to_csv("total_opt_flow.csv", index=False)


