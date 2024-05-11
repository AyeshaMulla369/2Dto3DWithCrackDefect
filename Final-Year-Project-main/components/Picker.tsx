/* eslint-disable quotes */
/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable no-trailing-spaces */
/* eslint-disable prettier/prettier */
import React, { useState, useEffect } from 'react';
import { Pressable, StyleSheet, Text, View, Image, PermissionsAndroid, Platform, ScrollView ,Alert, TouchableOpacity} from 'react-native';
import ImagePicker, { Image as ImageType } from 'react-native-image-crop-picker';
import axios from 'axios';
import { useNavigation } from '@react-navigation/native';

const Picker = () => {
  const [selectedImage, setSelectedImage] = useState<ImageType | null>(null);
  const [selectedImage2, setSelectedImage2] = useState<ImageType | null>(null);
  const [selectedImage3, setSelectedImage3] = useState<ImageType | null>(null);

  const [selectedImageUrl, setSelectedImageUrl] = useState('');
  const [selectedImageUrl2, setSelectedImageUrl2] = useState('');
  const [selectedImageUrl3, setSelectedImageUrl3] = useState('');
  const navigation = useNavigation();


  useEffect(() => {
    if (Platform.OS === 'android') {
      requestCameraPermission();
    }
  }, []);

  const requestCameraPermission = async () => {
    try {
      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.CAMERA,
        {
          title: 'Cool Photo App Camera Permission',
          message:
            'Cool Photo App needs access to your camera ' +
            'so you can take awesome pictures.',
          buttonNeutral: 'Ask Me Later',
          buttonNegative: 'Cancel',
          buttonPositive: 'OK',
        },
      );
      if (granted === PermissionsAndroid.RESULTS.GRANTED) {
        console.log('You can use the camera');
      } else {
        console.log('Camera permission denied');
      }
    } catch (err) {
      console.warn(err);
    }
  };


  const handleImageUpload = async () => {
    try {
      const image = await ImagePicker.openPicker({
        cropping: true,
      });
      console.log(image);
      setSelectedImage(image);
      const cloudUri = await handleUpload(image.path);
      setSelectedImageUrl(cloudUri);
    } catch (error) {
      console.log('ImagePicker Error: ', error);
    }
  };
  const handleImageUpload2 = async () => {
    try {
      const image = await ImagePicker.openPicker({
        cropping: true,
      });
      setSelectedImage2(image);
      const cloudUri = await handleUpload(image.path);
      setSelectedImageUrl2(cloudUri);
    } catch (error) {
      console.log('ImagePicker Error: ', error);
    }
  };
  const handleImageUpload3 = async () => {
    try {
      const image = await ImagePicker.openPicker({
        cropping: true,
      });
      setSelectedImage3(image);
      const cloudUri = await handleUpload(image.path);
      setSelectedImageUrl3(cloudUri);
    } catch (error) {
      console.log('ImagePicker Error: ', error);
    }
  };

  // const handleCameraUpload = async () => {
  //   try {
  //     const image = await ImagePicker.openCamera({
  //       cropping: true,
  //     });
  //     setSelectedImage(image);
  //   } catch (error) {
  //     console.log('ImagePicker Error: ', error);
  //   }
  // };




    
const handleUpload = async (post) => {
  if (post != null) {
    const formData = new FormData();
    formData.append('file', {
      uri: post,
      type: 'image/jpeg',
      name: 'uploaded_image',
    });
    formData.append('upload_preset', 'yivau9kc'); // Use your actual upload preset name

    try {
      const response = await axios.post('https://api.cloudinary.com/v1_1/dnz4gywty/image/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Upon successful upload, Cloudinary typically responds with the image URL
      const imageUrl = response.data.secure_url;
      console.log(imageUrl);

      return imageUrl;

    } catch (error) {
      // console.error('Error uploading image:', error);
      // console.log('Cloudinary Response:', error.response.data);
      Alert.alert('Please select an image for post.');
    }
  } else {
    Alert.alert('Please select an image');
  }
};


const getModel = async () => {
  try {
    console.log('front',selectedImageUrl);
    console.log("1ns", selectedImageUrl2);
    console.log("3rd", selectedImageUrl3);
    const res = await axios.get("http://192.168.29.190:5000/getModel", {
      params: {
        front: selectedImageUrl3,
        side: selectedImageUrl2,
        top: selectedImageUrl
      }
    });
    console.log(res);
    if (res.status === 200) {
      Alert.alert('Model created successfully');
      navigation.navigate('Model'); // Navigate to the main page or perform any other action
    } else {
      Alert.alert('Failed to create post');
    }
  } catch (err) {
    console.log(err);
  }
}
  

  return (

    <ScrollView>
    <View style={{flex:1,alignItems:'center',justifyContent:'center',gap:20,padding:20,backgroundColor:'#f8f4ff',marginTop:40}}>
   
   <Pressable
     style={{
       backgroundColor: '#4169e1',
       borderColor: '#C0C0C0',
       borderWidth: 1,
       width: '80%',
       borderRadius: 10,
       padding: 12,
       alignItems: 'center',
       justifyContent: 'center',
       shadowColor: '#000',
       shadowOffset: {
         width: 0,
         height: 3,
       },
       shadowOpacity: 0.27,
       shadowRadius: 4.65,
       elevation: 6,
       marginBottom: 20,
     }}
     onPress={handleImageUpload}>
     <Text style={{ fontSize: 16, color: '#fff' }}>Top View</Text>
   </Pressable>
   <Pressable
     style={{
       backgroundColor: '#4169e1',
       borderColor: '#C0C0C0',
       borderWidth: 1,
       width: '80%',
       borderRadius: 10,
       padding: 12,
       alignItems: 'center',
       justifyContent: 'center',
       shadowColor: '#000',
       shadowOffset: {
         width: 0,
         height: 3,
       },
       shadowOpacity: 0.27,
       shadowRadius: 4.65,
       elevation: 6,
       marginBottom: 20,
     }}
     onPress={handleImageUpload2}>
     <Text style={{ fontSize: 16, color: '#fff' }}>Side View</Text>
   </Pressable>
   <Pressable
     style={{
       backgroundColor: '#4169e1',
       borderColor: '#C0C0C0',
       borderWidth: 1,
       width: '80%',
       borderRadius: 10,
       padding: 12,
       alignItems: 'center',
       justifyContent: 'center',
       shadowColor: '#000',
       shadowOffset: {
         width: 0,
         height: 3,
       },
       shadowOpacity: 0.27,
       shadowRadius: 4.65,
       elevation: 6,
       marginBottom: 20,
     }}
     onPress={handleImageUpload3}>
     <Text style={{ fontSize: 16, color: '#fff' }}>Front View</Text>
   </Pressable>

   {/* <Pressable
     style={{
       backgroundColor: '#4169e1',
       borderColor: '#C0C0C0',
       borderWidth: 1,
       width: '80%',
       borderRadius: 10,
       padding: 12,
       alignItems: 'center',
       justifyContent: 'center',
       shadowColor: '#000',
       shadowOffset: {
         width: 0,
         height: 3,
       },
       shadowOpacity: 0.27,
       shadowRadius: 4.65,
       elevation: 6,
     }}
     onPress={handleCameraUpload}>
     <Text style={{ fontSize: 16, color: '#fff' }}>Take a Photo</Text>
   </Pressable> */}
   
   {selectedImage && (
     <View
       style={{
         alignItems: 'center',
         marginTop: 20,
         borderRadius: 20,
         backgroundColor: 'white',
         shadowColor: '#000', 
         shadowOffset: { width: 0, height: 4 },
         shadowOpacity: 0.3,
         shadowRadius: 4,
         elevation: 8,
       }}>
         <Text style={{color:'#1f75fe',fontSize:18,}}>Top</Text>
       <Image
         source={{ uri: selectedImage.path }}
         style={{
           width: 300,
           height: 220,
           borderRadius: 20,
           alignSelf: 'center',
         }}
         resizeMode="cover"
         alt="Selected Image"
       />
     </View>
   )}
   {selectedImage2 && (
     <View
       style={{
         alignItems: 'center',
         marginTop: 20,
         borderRadius: 20,
         backgroundColor: 'white',
         shadowColor: '#000', 
         shadowOffset: { width: 0, height: 4 },
         shadowOpacity: 0.3,
         shadowRadius: 4,
         elevation: 8,
       }}>
         <Text style={{color:'#1f75fe',fontSize:18,}}>Side</Text>
       <Image
         source={{ uri: selectedImage2.path }}
         style={{
           width: 300,
           height: 220,
           borderRadius: 20,
           alignSelf: 'center',
         }}
         resizeMode="cover"
         alt="Selected Image"
       />
     </View>
   )}
   {selectedImage3 && (
     <View
       style={{
         alignItems: 'center',
         marginTop: 20,
         borderRadius: 20,
         backgroundColor: 'white',
         shadowColor: '#000', 
         shadowOffset: { width: 0, height: 4 },
         shadowOpacity: 0.3,
         shadowRadius: 4,
         elevation: 8,
       }}>
         <Text style={{color:'#1f75fe',fontSize:18,}}>Bottom</Text>
       <Image
         source={{ uri: selectedImage3.path }}
         style={{
           width: 300,
           height: 220,
           borderRadius: 20,
           alignSelf: 'center',
         }}
         resizeMode="cover"
         alt="Selected Image"
       />
     </View>
   )}

    <TouchableOpacity
      onPress={getModel}
    >
      <Text>Get model</Text>
    </TouchableOpacity>


 </View>
 </ScrollView>
   
  );
};

export default Picker;

const styles = StyleSheet.create({});
