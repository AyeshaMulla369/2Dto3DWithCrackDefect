/* eslint-disable prettier/prettier */
/* eslint-disable eol-last */
/* eslint-disable no-alert */
/* eslint-disable no-trailing-spaces */
/* eslint-disable no-unused-vars */
/* eslint-disable semi */
import React, { useState } from 'react';
import { StyleSheet, Text, useWindowDimensions } from 'react-native';
import { TabBar, TabView } from 'react-native-tab-view';
import Picker from './Picker';
import CrackImagePicker from './CrackImagePicker';

export default function TabsScreen({ navigation }) {
  const layout = useWindowDimensions();
  const [index, setIndex] = useState(0);
  const [routes] = useState([
    { key: 'twoD-tab', title: '2D to 3D' },
    { key: 'crack-tab', title: 'Defect Analysis' },
  ]);
  

  const renderScene = ({ route }) => {
    switch (route.key) {
      case 'twoD-tab':
        return <Picker/>;
      case 'crack-tab':
        return <CrackImagePicker/>;
      default:
        return null;
    }
  };

  const renderTabBar = props => (
    <TabBar
      {...props}
      renderLabel={scene => (
        <Text
          style={[
            styles.label,
            {
              color: scene.focused ? "#000" : "rgba(59,72,89,0.8)",
            },
          ]}>
          {scene.route.title}
        </Text>
      )}
      style={styles.tabBar}
      indicatorStyle={styles.indicator}
    />
  );

  return (
    <>
      <TabView
        useNativeDriver
        navigationState={{ index, routes }}
        renderScene={renderScene}
        onIndexChange={setIndex}
        initialLayout={{ width: layout.width }}
        renderTabBar={renderTabBar}
        sceneContainerStyle={{
          width:'100%'
        }}
      />
    </>

  );
}

const styles = StyleSheet.create({
  label: {
    fontFamily: 'Lexend-Medium',
    fontSize: 15,
  },
  indicator: {
    backgroundColor: "#000",
    height: 1.5,
  },
  tabBar: {
    backgroundColor: 'transparent',
  },
});
