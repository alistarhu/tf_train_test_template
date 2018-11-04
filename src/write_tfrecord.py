import os
import tfrecord_write_read as tfwr

# test for write tfrecord demo
img_dir = '/media/data1/hl/expression_cls_data/expression_dect_data/expression_crop64x64_dataArgu'
img_list_dir = '/media/data1/hl/expression_cls_data/expression_dect_data/expression_crop64x64_dataArgu'
output_dir = '/media/data1/hl/expression_cls_data/expression_dect_tfrecords64x64'
file_list = ["train", "valid", "test"]

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
f = open(output_dir+'/record_count.txt', 'w')
for ii in file_list:
    record_count = 0
    out_file = os.path.join(output_dir, 'expression_dect' + '_{}.tfrecord'.format(ii))
    print out_file
    record_count = tfwr.write_tfrecord(img_list_dir+"/{}_list.txt".format(ii), img_dir, out_file)
    f.write('The {} count is: {}\n'.format(ii, record_count))
f.close()
