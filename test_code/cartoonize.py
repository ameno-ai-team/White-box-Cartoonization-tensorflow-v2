import os
import cv2
import numpy as np

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import network
import guided_filter
from tqdm import tqdm

# Disable eager execution
tf.compat.v1.disable_eager_execution()




def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image
    

def cartoonize(load_folder, save_folder, model_path):
    print("model_path",model_path)
    input_photo = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.compat.v1.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.compat.v1.train.Saver(var_list=gene_vars)
    
    # Use CPU only
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
    sess = tf.compat.v1.Session(config=config)
    print("Using CPU for processing")

    sess.run(tf.compat.v1.global_variables_initializer())
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint(model_path))
    name_list = os.listdir(load_folder)
    for name in tqdm(name_list):
        try:
            load_path = os.path.join(load_folder, name)
            save_path = os.path.join(save_folder, name)
            
            # Check if input file exists
            if not os.path.exists(load_path):
                print(f'Error: Input file {load_path} does not exist')
                continue
                
            # Load and preprocess image
            image = cv2.imread(load_path)
            if image is None:
                print(f'Error: Could not read image {load_path}')
                continue
                
            image = resize_crop(image)
            batch_image = image.astype(np.float32)/127.5 - 1
            batch_image = np.expand_dims(batch_image, axis=0)
            
            try:
                # Run cartoonization
                output = sess.run(final_out, feed_dict={input_photo: batch_image})
                output = (np.squeeze(output)+1)*127.5
                output = np.clip(output, 0, 255).astype(np.uint8)
                
                # Save output
                cv2.imwrite(save_path, output)
                print(f'Successfully cartoonized {name}')
            except Exception as e:
                print(f'Error cartoonizing {name}: {str(e)}')
        except Exception as e:
            print(f'Failed to process {name}: {str(e)}')


    

if __name__ == '__main__':
    model_path = 'saved_models'
    load_folder = 'test_images'
    save_folder = 'cartoonized_images'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    cartoonize(load_folder, save_folder, model_path)
    

    