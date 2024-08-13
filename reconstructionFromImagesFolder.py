import os
import glob
import numpy as np
import open3d as o3d
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation, DPTImageProcessor, DPTForDepthEstimation
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ['OMP_NUM_THREADS'] = '1'


def process_image(image_path, feature_extractor, model):
    image = Image.open(image_path)
    new_height = 960 if image.height > 960 else image.height
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    new_image_size = (new_width, new_height)
    image = image.resize(new_image_size)

    inputs = feature_extractor(images=image, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    pad = 16
    output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    output = output[pad: -pad, pad: -pad]
    image = image.crop((pad, pad, image.width - pad, image.height - pad))

    # Debug: Visualize depth map
    # plt.imshow(output, cmap='inferno')
    # plt.title("Depth Map")
    # plt.show()

    return image, output


def create_point_cloud(image, depth, camera_intrinsic):
    # Debug: Check the range and distribution of depth values
    # print(f"Depth map min: {depth.min()}, max: {depth.max()}, mean: {depth.mean()}")

    # Normalize depth for visualization
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    depth_o3d = o3d.geometry.Image((depth_normalized * 255).astype('uint8'))
    image_o3d = o3d.geometry.Image(np.array(image))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    # Debug: Check the number of points
    # print(f"Point Cloud has {len(pcd.points)} points before rotation.")
    return pcd


def rotate_point_cloud(pcd):
    R = pcd.get_rotation_matrix_from_xyz((0, np.pi, np.pi))  # Rotate 180 degrees around y and z axes
    pcd.rotate(R, center=(0, 0, 0))
    return pcd


def save_point_cloud(pcd, directory, index):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f"point_cloud_{index}.ply")
    o3d.io.write_point_cloud(file_path, pcd, write_ascii=True)  # Save as ASCII format
    return file_path


def load_point_clouds(file_paths):
    point_clouds = []
    for file_path in file_paths:
        pcd = o3d.io.read_point_cloud(file_path)
        point_clouds.append(pcd)
    return point_clouds


def combine_point_clouds(file_paths, output_path):
    combined_pcd = o3d.geometry.PointCloud()
    for file_path in tqdm(file_paths, desc="Combining Point Clouds"):
        pcd = o3d.io.read_point_cloud(file_path)
        combined_pcd += pcd

    print(f"Combined point cloud has {len(combined_pcd.points)} points.")
    o3d.io.write_point_cloud(output_path, combined_pcd, write_ascii=True)  # Save as ASCII format
    return combined_pcd


def main(image_dir,isNew = 0, batch_size=100):
    saved_dir = '/Users/leon/M&L/ImagesTo3DObject/pythonProject/tmpFiles'
    combined_dir = '/Users/leon/M&L/ImagesTo3DObject/pythonProject/combinedFiles'

    # To save time on image processing I save the copy of the data
    if isNew == 0:
        print("Saved point clouds found. Loading from disk...")
        saved_paths = glob.glob(os.path.join(saved_dir, '*.ply'))
    else:
        # Iterate through all files in the folder
        for filename in os.listdir(saved_dir):
            file_path = os.path.join(saved_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete the file
                print(f"Deleted: {file_path}")
        for filename in os.listdir(combined_dir):
            file_path = os.path.join(combined_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete the file
                print(f"Deleted: {file_path}")
        feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
        model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
        # feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        # model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

        image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))

        for i, image_path in enumerate(tqdm(image_paths, desc="Processing Images")):
            image, depth = process_image(image_path, feature_extractor, model)

            width, height = image.size
            camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
            camera_intrinsic.set_intrinsics(width, height, 500, 500, width / 2, height / 2)

            pcd = create_point_cloud(image, depth, camera_intrinsic)
            pcd = rotate_point_cloud(pcd)

            save_point_cloud(pcd, saved_dir, i)

    saved_paths = glob.glob(os.path.join(saved_dir, '*.ply'))
    print(f"Point clouds saved to {saved_dir}")

    combined_paths = []
    for i in range(0, len(saved_paths), batch_size):
        batch_paths = saved_paths[i:i + batch_size]
        try:
            combined_pcd = combine_point_clouds(batch_paths,
                                                os.path.join(combined_dir, f"combined_batch_{i // batch_size}.ply"))
            combined_paths.append(os.path.join(combined_dir, f"combined_batch_{i // batch_size}.ply"))
        except Exception as e:
            print(f"Failed to write combined point cloud batch {i // batch_size}: {e}")

    try:
        combined_pcd = combine_point_clouds(combined_paths, "combined_point_cloud.ply")
    except Exception as e:
        print(f"Failed to write final combined point cloud: {e}")
    voxel_size = 1e-05
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"After downsampling: {len(combined_pcd.points)} points.")

    if len(combined_pcd.points) > 0:
        cl, ind = combined_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=5.0)
        combined_pcd = combined_pcd.select_by_index(ind)
        print(f"After outlier removal: {len(combined_pcd.points)} points.")

        if len(combined_pcd.points) > 0:
            combined_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            if np.asarray(combined_pcd.normals).shape[0] > 0:
                print("Normals estimated successfully.")
                combined_pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

                o3d.visualization.draw_geometries([combined_pcd])

                try:
                    o3d.io.write_point_cloud("combined_point_cloud_final.ply", combined_pcd, write_ascii=True)
                except Exception as e:
                    print(f"Failed to write final combined point cloud: {e}")
            else:
                print("Normal estimation failed. The point cloud might be too sparse.")
        else:
            print("After outlier removal, no points remain. Try adjusting the outlier removal parameters.")
    else:
        print("After downsampling, no points remain. Try using a smaller voxel size.")


if __name__ == "__main__":
    image_dir = 'images/minion'
    # image_dir = '../images/flower'
    main(image_dir, 1)
