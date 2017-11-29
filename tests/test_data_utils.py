from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))
import utils.ubuntu_data_utils as du
import numpy as np

def test_get_embeddings_tensor():
    embeddings_tensor, word_to_indx = du.get_embeddings_tensor()
    print "Size embeddings tensor: {}".format(embeddings_tensor.shape)
    user_indx = word_to_indx['user']
    user_embedding = embeddings_tensor[user_indx,:]
    print "Embedding for ""user"":\n {}".format(user_embedding)
    assert np.equal(user_embedding,np.array(map(np.float32,"0.088813 -0.004539 -0.008035 -0.021437 0.060462 0.007162 0.153963 -0.045462 0.022156 -0.034045 0.046460 -0.107129 -0.043072 0.074368 0.120642 -0.102444 0.079050 -0.017929 -0.012550 0.034706 0.028370 0.093264 0.021148 -0.078800 -0.083400 0.048900 -0.038533 -0.070483 -0.082696 -0.025121 -0.106042 -0.083764 0.107170 -0.116584 -0.098069 0.016174 0.096907 0.060335 -0.073326 0.036980 0.025710 0.064141 0.144856 0.121593 -0.025466 0.008252 0.009652 -0.040288 -0.139219 -0.040576 0.119039 0.010055 -0.034774 0.091904 -0.121002 0.073930 -0.044172 0.002382 -0.031404 -0.041262 0.026838 0.119991 0.028032 -0.033010 0.053610 -0.008170 -0.049982 0.013084 0.016903 -0.119127 0.018084 -0.077641 -0.159491 -0.074125 0.007174 -0.116886 -0.083645 -0.004723 -0.052470 -0.049358 -0.043767 0.129888 -0.084597 0.034285 -0.185354 -0.021527 0.081748 0.044528 0.040812 -0.015750 0.117037 -0.031904 -0.099353 0.112485 -0.008626 0.061718 -0.140435 -0.003587 0.045944 0.040946 0.176409 -0.007869 -0.018806 0.018175 -0.051294 -0.114214 -0.030233 -0.058147 -0.027051 -0.019574 -0.121775 0.148758 0.004262 0.008878 -0.037327 -0.065138 0.062889 0.042872 -0.090468 -0.019457 -0.021442 -0.013768 0.031753 -0.097717 -0.016655 -0.058313 -0.025982 0.001182 -0.035869 0.032188 -0.057690 -0.048517 0.001181 -0.163698 -0.033952 0.028203 -0.060373 0.064387 -0.105128 0.027086 0.001070 0.018298 0.085190 0.053762 0.030682 0.001386 -0.026149 -0.092223 0.047697 -0.019690 0.038803 0.106383 0.017631 -0.020903 0.161101 -0.039422 -0.019532 0.111115 0.022723 0.081487 0.109038 -0.015792 -0.040997 0.016530 0.091677 0.090276 -0.099224 -0.036687 -0.036072 0.069073 -0.040577 0.056626 -0.083551 -0.031621 -0.012924 0.024927 -0.016690 -0.086189 0.035489 -0.012425 -0.095727 0.015795 -0.038825 -0.075390 -0.000305 0.098599 0.110577 0.043111 -0.044103 0.106845 0.073305 -0.088903 0.057435 -0.011955 0.034989 0.090531 0.079599 -0.113229 0.061071 0.065560 ".strip().split(" ")))).all()


def test_get_data_dict():
    data_dict = du.get_data_dict()
    print "Length data dict: {}".format(len(data_dict))

    title, body = data_dict[315984]
    print "Title for qid 315984: {}".format(title)
    print "Body for qid 315984: {}".format(body)
    assert title == "mysql-client 5.6 ubuntu 12.04"
    assert body == "how long does ubuntu us usually take to release official packages for newly release software ? such as mysql-client 5.6"

    title, body = data_dict[23563]
    print "Title for qid 23563: {}".format(title)
    print "Body for qid 23563: {}".format(body)
    assert title == "why is `` shutdown p 01:00 '' not working ?"
    assert body == ""

def test_get_train_dataset():
    train_data = du.get_train_examples()
    print "Number of train data: {}".format(len(train_data))
    first_example = train_data[0]
    print "First example: {}".format(first_example)

    assert first_example[0] == 262144
    assert first_example[1] == [211039]
    assert first_example[2] == map(int,"227387 413633 113297 356390 256881 145638 296272 318659 86529 48563 53080 65996 334032 517236 470002 177348 502185 248772 457062 339049 265060 264524 15971 40468 422021 177982 513266 437762 318889 163711 491359 72668 523366 513514 119253 167106 395661 411198 284222 75399 352479 250977 476895 384032 379657 47506 214327 295133 165345 278645 40986 147959 498308 328292 49304 518817 418947 155658 522902 316447 487482 100000 436018 235318 314941 187592 439142 26205 180583 418425 41667 188060 147132 68407 191082 455272 486933 229959 153150 101914 254265 314232 384593 366741 227996 384171 302283 335076 452240 144708 395995 221644 392821 197651 404031 319280 257320 38398 287787 83851".split(" "))

def test_get_dev_dataset():
    dev_data = du.get_dev_examples()
    print "Number of dev data: {}".format(len(dev_data))
    first_example = dev_data[0]
    print "First example: {}".format(first_example)
    assert len(dev_data) == 200

def test_get_test_dataset():
    test_data = du.get_test_examples()
    print "Number of test data: {}".format(len(test_data))
    first_example = test_data[0]
    print "First example: {}".format(first_example)
    assert len(test_data) == 200

if __name__ == "__main__":
    test_get_embeddings_tensor()
    test_get_data_dict()
    test_get_train_dataset()
    test_get_dev_dataset()
    test_get_test_dataset()
    print "\nSUCCESS: All tests passed."
