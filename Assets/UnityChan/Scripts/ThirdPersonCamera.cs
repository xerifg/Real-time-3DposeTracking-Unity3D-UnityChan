//
// Unityちゃん用の三人称カメラ
// 
// 2013/06/07 N.Kobyasahi
//
using UnityEngine;
using System.Collections;


public class ThirdPersonCamera : MonoBehaviour
{
	public float smooth = 3f;		// 用于平滑相机运动的变量
	Transform standardPos;			// the usual position for the camera, specified by a transform in the game
	Transform frontPos;			// Front Camera locater
	Transform jumpPos;			// Jump Camera locater
	
	// 连接不顺畅时的布尔标志（快速切换）
	bool bQuickSwitch = false;	//Change Camera Position Quickly
	
	
	void Start()
	{
		// 各参照の初期化
		standardPos = GameObject.Find ("FrontPos").transform;
		
		if(GameObject.Find ("BackPos"))
			frontPos = GameObject.Find ("BackPos").transform;

		if(GameObject.Find ("NearPos"))
			jumpPos = GameObject.Find ("NearPos").transform;

		//カメラをスタートする
			transform.position = standardPos.position;	
			transform.forward = standardPos.forward;	
	}

	
	void FixedUpdate ()	// 除非在FixedUpdate()中，否则此相机切换无法正常工作
	{
		
		if(Input.GetButton("Fire1"))	// left Ctlr
		{	
			// Change Front Camera
			setCameraPositionFrontView();
		}
		
		else if(Input.GetButton("Fire2"))	//Alt
		{	
			// Change to near Camera
			setCameraPositionNearView();
		}
		
		else
		{	
			// return the camera to standard position and direction
			setCameraPositionNormalView();
		}
	}

	void setCameraPositionNormalView()
	{
		if(bQuickSwitch == false){
		// the camera to standard position and direction
						transform.position = Vector3.Lerp(transform.position, standardPos.position, Time.fixedDeltaTime * smooth);	
						transform.forward = Vector3.Lerp(transform.forward, standardPos.forward, Time.fixedDeltaTime * smooth);
		}
		else{
			// the camera to standard position and direction / Quick Change
			transform.position = standardPos.position;	
			transform.forward = standardPos.forward;
			bQuickSwitch = false;
		}
	}

	
	void setCameraPositionFrontView()
	{
		// Change Front Camera
		bQuickSwitch = true;
		transform.position = frontPos.position;	
		transform.forward = frontPos.forward;
	}

	void setCameraPositionNearView()
	{
		// Change to near Camera
		bQuickSwitch = false;
		transform.position = Vector3.Lerp(transform.position, jumpPos.position, Time.fixedDeltaTime * smooth);	
		transform.forward = Vector3.Lerp(transform.forward, jumpPos.forward, Time.fixedDeltaTime * smooth);		
	}
}
