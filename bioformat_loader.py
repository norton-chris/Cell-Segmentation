
import javabridge
import bioformats as bf
import numpy as np

filename='TrainingDataset/data_subset/output/test/Images/393.tif'                                                                                                                      #file path for bioformat image/movie
try:
    print('\nStarting Java Bridge \n')
    javabridge.start_vm(class_path=bf.JARS, run_headless=True)                                                                                  #starts the javabridge


    #suppress the java debug messages :: https://forum.image.sc/t/python-bioformats-and-javabridge-debug-messages/12578/11
    rootLoggerName = javabridge.get_static_field("org/slf4j/Logger","ROOT_LOGGER_NAME", "Ljava/lang/String;")
    rootLogger = javabridge.static_call("org/slf4j/LoggerFactory","getLogger", "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
    logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level","ERROR", "Lch/qos/logback/classic/Level;")
    javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)


    r=bf.ImageReader(filename,perform_init=True)                                                                                                #creates reader

    meta=bf.OMEXML(bf.get_omexml_metadata(filename))                                                                                            #retrieves metadata from file

    print("\nLoaded Java Bridge & Reader\n")

    c=meta.image().Pixels.SizeC                                                                                                                 #num of channels
    z=meta.image().Pixels.SizeZ                                                                                                                 #num of z slices, 1 if 2d image
    t=meta.image().Pixels.SizeT                                                                                                                 #num of time intervals 
    n=c*z*t                                                                                                                                     #total dimension of file
    print("Total Images :: {} \n\tTime Intervals :: {} \n\tZ Planes :: {} \n\tChannels :: {}".format(n,t,z,c))

    for tt in range(t):
        for zz in range(z):
            for cc in range(c):
                print('Image {} of {}'.format((tt*z*c+zz*c+cc),n))
                im=r.read(c=cc,z=zz,t=tt)                                                                                                       #reads the image at time tt at z slice zz in channel cc as an array.  
                




finally:
    print("Killing Bridge")
    javabridge.kill_vm()                                                                                                                        #closes java vm


