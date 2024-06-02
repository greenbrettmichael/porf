import pickle
from path import Path
import numpy as np
import sqlite3
import os


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2


def image_ids_to_pair(image_id1, image_id2):
    pair_id = image_id2 + 2147483647 * image_id1
    return pair_id


def get_keypoints(cursor, image_id):
    cursor.execute("SELECT * FROM keypoints WHERE image_id = ?;", (image_id,))
    image_idx, n_rows, n_columns, raw_data = cursor.fetchone()
    kypnts = np.frombuffer(raw_data, dtype=np.float32).reshape(n_rows, n_columns).copy()
    kypnts = kypnts[:, 0:2]
    return kypnts

def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def parse_colmap(filename):
    return read_pickle(filename)

def process_one_scene(scene_dir):

    filename_db = Path(scene_dir)/'database.db'
    outdir = scene_dir/'colmap_matches'
    print("Opening database: " + filename_db)

    if not os.path.exists(filename_db):
        print('Error db does not exist!')
        exit()

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    print(f'Clean old matches in {outdir}')
    for f in Path(outdir).files('*'):
        os.remove(f)

    connection = sqlite3.connect(filename_db)
    cursor = connection.cursor()

    list_image_ids = []

    img_ids_to_names_dict = {}
    db_id_to_image_id = {}
    sequence_id_to_image_id = {}
    db_id_to_sequence_id = {}
    poses, Ks, image_names, img_ids, center, scale, directions =parse_colmap(Path(scene_dir)/'cache.pkl')

    cursor.execute('SELECT image_id, name, cameras.width, cameras.height FROM images LEFT JOIN cameras ON images.camera_id == cameras.camera_id;')
    for row in cursor:
        image_idx, name, width, height = row
        if image_idx not in img_ids:
            db_id_to_image_id[image_idx] = -1
            continue
        list_image_ids.append(image_idx)
        img_ids_to_names_dict[image_idx] = name
        db_id_to_image_id[image_idx] = image_idx
        sequence_id = len(list_image_ids)-1
        sequence_id_to_image_id[sequence_id] = image_idx
        db_id_to_sequence_id[image_idx] = sequence_id

    num_image_ids = len(list_image_ids)
    
    camera_dict = {}
    for idx in range(num_image_ids):
        img_id = sequence_id_to_image_id[idx]
        camera_dict['world_mat_%d' % idx] = poses[img_id]
        camera_dict['scale_mat_%d' % idx] = np.eye(4) * scale
    np.savez(Path(scene_dir)/'cameras_colmap.npz', **camera_dict)

    

    # Iterate over entries in the two-view geometry table
    cursor.execute('SELECT pair_id, rows, cols, data FROM two_view_geometries;')
    all_matches = {}
    for row in cursor:
        pair_id = row[0]
        rows = row[1]
        cols = row[2]
        raw_data = row[3]
        if (rows < 5):
            continue

        matches = np.frombuffer(raw_data, dtype=np.uint32).reshape(rows, cols)

        if matches.shape[0] < 5:
            continue

        all_matches[pair_id] = matches

    for key in all_matches:
        pair_id = key
        matches = all_matches[key]

        # # skip if too few matches are given
        # if matches.shape[0] < 300:
        #     continue

        id1, id2 = pair_id_to_image_ids(pair_id)
        if db_id_to_image_id[id1] == -1 or db_id_to_image_id[id2] == -1:
            continue
        
        keys1 = get_keypoints(cursor, id1)
        keys2 = get_keypoints(cursor, id2)

        id1 = db_id_to_sequence_id[id1]+1
        id2 = db_id_to_sequence_id[id2]+1

        match_positions = np.empty([matches.shape[0], 4])
        for i in range(0, matches.shape[0]):
            match_positions[i, :] = np.array([keys1[matches[i, 0]][0], keys1[matches[i, 0]][1], keys2[matches[i, 1]][0], keys2[matches[i, 1]][1]])

        outfile = os.path.join(outdir, '{:06d}_{:06d}.txt'.format(int(id1), int(id2)))

        np.savetxt(outfile, match_positions, delimiter=' ')

        # reverse and save
        match_positions_reverse = np.concatenate([match_positions[:, 2:4], match_positions[:, 0:2]], axis=1)
        outfile = os.path.join(outdir, '{:06d}_{:06d}.txt'.format(int(id2), int(id1)))
        np.savetxt(outfile, match_positions_reverse, delimiter=' ')

    cursor.close()
    connection.close()

    for idx in range(num_image_ids):

        two_view = {}
        two_view["src_idx"] = []
        two_view["match"] = []

        files = sorted(outdir.files('{:06d}_*.txt'.format(idx+1)))
        for f in files:
            j = int(os.path.basename(f)[7:13])-1

            one_pair = np.loadtxt(f)

            two_view["src_idx"].append(j)
            two_view["match"].append(one_pair)
        
        with open(outdir/'{:06d}.pkl'.format(idx), 'wb') as f:
            pickle.dump(two_view, f)


if __name__ == "__main__":

    data_dir = Path("C:/Users/me/Downloads/fender_6_2/processed")
    process_one_scene(scene_dir=data_dir)
