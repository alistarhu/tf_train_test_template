import os
import tfrecord_write_read as tfwr

# test for write tfrecord demo
img_dir = '/media/data1/hl/expression_cls_data/expression_dect_data/expression_test_crop64x64/data'
img_list_dir = '/media/data1/hl/expression_cls_data/expression_dect_data/expression_test_crop64x64/data'
output_dir = '/media/data1/hl/expression_cls_data/expression_dect_tfrecords64x64'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
f = open(output_dir+'/real_test_record_count.txt', 'w')
record_count = 0
out_file = os.path.join(output_dir, 'expression_dect' + '_{}.tfrecord'.format('real_test'))
print out_file
record_count = tfwr.write_tfrecord(img_list_dir+"/real_test_list.txt", img_dir, out_file)
f.write('The count is: {}\n'.format(record_count))
f.close()
