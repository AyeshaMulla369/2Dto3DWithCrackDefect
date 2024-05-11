import React, { useEffect, useState } from 'react';
import { View, Text, Image } from 'react-native';

const CardComponent = ({ imageUri, title }) => {
  return (
    <View style={{
      backgroundColor: '#fff',
      borderRadius: 8,
      padding: 16,
      shadowColor: '#000',
      shadowOffset: {
        width: 0,
        height: 2,
      },
      shadowOpacity: 0.25,
      shadowRadius: 3.84,
      elevation: 5,
      alignItems: 'center',
      margin: 16,
    }}>
      <Text style={{
        fontSize: 18,
        fontWeight: 'bold',
        marginBottom: 8,
      }}>{title}</Text>
      <Image source={{ uri: imageUri }} style={{
        width: 300,
        height: 300,
        borderRadius: 8,
        marginTop: 8,
      }} />
      <Text style={{
        fontSize: 14,
        color: '#888',
        marginTop: 8,
      }}>Selected Image of Crack</Text>
    </View>
  );
};

export default CardComponent;
