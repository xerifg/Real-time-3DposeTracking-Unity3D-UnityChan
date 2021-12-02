//
// Mecanimのアニメーションデータが、原点で移動しない場合の Rigidbody付きコントローラ
// サンプル
// 2014/03/13 N.Kobyasahi
//
using UnityEngine;
using System.Collections;

// 所需组件列表
[RequireComponent(typeof (Animator))]
[RequireComponent(typeof (CapsuleCollider))]
[RequireComponent(typeof (Rigidbody))]

public class UnityChanControlScriptWithRgidBody : MonoBehaviour
{

	public float animSpeed = 1.5f;				// 动画播放速度设置
	public float lookSmoother = 3.0f;			// a smoothing setting for camera motion
	public bool useCurves = true;				// 在 Mecanim 中使用或设置曲线调整
												// 如果这个开关没有打开，曲线将不会被使用
	public float useCurvesHeight = 0.5f;		// 曲线矫正有效高度（易滑过地面时增加）

	// 字符控制器的以下参数
	// 前進速度
	public float forwardSpeed = 7.0f;
	// 後退速度
	public float backwardSpeed = 2.0f;
	// 旋回速度
	public float rotateSpeed = 2.0f;
	// 跳跃力
	public float jumpPower = 3.0f; 
	// 角色控制器参考（胶囊对撞机）
	private CapsuleCollider col;
	private Rigidbody rb;
	// 角色控制器的移动量（胶囊碰撞器）
	private Vector3 velocity;
	// 存储CapsuleCollider中设置的碰撞器Heiht和Center的初始值的变量
	private float orgColHight;
	private Vector3 orgVectColCenter;
	
	private Animator anim;							// 对附加到角色的动画师的引用
	private AnimatorStateInfo currentBaseState;			// 对基础层中使用的动画师当前状态的引用

	private GameObject cameraObject;	// 参考主摄像头
		
// 动画师参考每个状态
	static int idleState = Animator.StringToHash("Base Layer.Idle");
	static int locoState = Animator.StringToHash("Base Layer.Locomotion");
	static int jumpState = Animator.StringToHash("Base Layer.Jump");
	static int restState = Animator.StringToHash("Base Layer.Rest");

// 初期化
	void Start ()
	{
		// 获取 Animator 组件
		anim = GetComponent<Animator>();
		// CapsuleCollider获取组件（胶囊型碰撞）
		col = GetComponent<CapsuleCollider>();
		rb = GetComponent<Rigidbody>();
		//获取主摄像头
		cameraObject = GameObject.FindWithTag("MainCamera");
		// CapsuleCollider保存组件的Height和Center的初始值
		orgColHight = col.height;
		orgVectColCenter = col.center;
}
	
	
// 下面是主要流程，由于是和刚体纠缠在一起，所以在Fixed Update中处理.
	void FixedUpdate ()
	{
		float h = Input.GetAxis("Horizontal");				// 用 h 定义输入设备的水平轴,检测水平方向键
		float v = Input.GetAxis("Vertical");				// 用 v 定义输入设备的垂直轴
		anim.SetFloat("Speed", v);							// Animator将 v 传递给侧面设置的“Speed”参数
		anim.SetFloat("Direction", h); 						// Animator将 h 传递给侧面设置的“Direction”参数
		anim.speed = animSpeed;								// Animator为动画播放速度设置animSpeed
		currentBaseState = anim.GetCurrentAnimatorStateInfo(0);	// 将参考状态变量设置为 Base Layer (0) 的当前状态
		rb.useGravity = true;//跳跃时重力被切断，否则受重力影响
		
		
		
		// 人物的运动过程
		velocity = new Vector3(0, 0, v);		// 从上下键输入获取Z轴方向的移动量
		// 转换为字符在局部空间中的方向
		velocity = transform.TransformDirection(velocity);
		//以下 v 阈值通过 Mecanim 端的转换进行调整
		if (v > 0.1) {
			velocity *= forwardSpeed;		// 移動速度を掛ける
		} else if (v < -0.1) {
			velocity *= backwardSpeed;	// 移動速度を掛ける
		}
		
		if (Input.GetButtonDown("Jump")) {	// 输入空格键后

			//动画状态只能在运动期间跳跃
			if (currentBaseState.fullPathHash == locoState){
				//如果您不在状态转换中，则可以跳转
				if(!anim.IsInTransition(0))
				{
						rb.AddForce(Vector3.up * jumpPower, ForceMode.VelocityChange);
						anim.SetBool("Jump", true);		// 向Animator发送一个标志以切换到跳转
				}
			}
		}
		

		// 按上下键移动字符
		transform.localPosition += velocity * Time.fixedDeltaTime;

		// 按左右键在Y轴上摆动字符
		transform.Rotate(0, h * rotateSpeed, 0);	
	

		// 下面，Animator在各个状态下的处理
		// Locomotion中
		// 当前基础层为 locoState 时
		if (currentBaseState.fullPathHash == locoState){
			//如果您要调整曲线上的碰撞器，请重置它以防万一
			if(useCurves){
				resetCollider();
			}
		}
		// JUMP中の処理
		// 当前基础层为 jumpState 时
		else if(currentBaseState.fullPathHash == jumpState)
		{
			cameraObject.SendMessage("setCameraPositionJumpView");	// 换成跳跃相机
			// 如果状态不在转换中
			if(!anim.IsInTransition(0))
			{
				
				// 以下、调整曲线时的处理
				if(useCurves){
					// 下面是附加到 JUMP00 动画的曲线跳跃高度和重力控制。
					// JumpHeight:JUMP00 处的跳跃高度（0〜1）
					// GravityControl:1⇒ジャンプ中（重力無効）、0⇒重力有効
					float jumpHeight = anim.GetFloat("JumpHeight");
					float gravityControl = anim.GetFloat("GravityControl"); 
					if(gravityControl > 0)
						rb.useGravity = false;	//ジャンプ中の重力の影響を切る
										
					// 从角色的中心放下光线投射
					Ray ray = new Ray(transform.position + Vector3.up, -Vector3.up);
					RaycastHit hitInfo = new RaycastHit();
					// 仅当高度高于 useCurvesHeight 时，使用附加到 JUMP00 动画的曲线调整碰撞器的高度和中心。
					if (Physics.Raycast(ray, out hitInfo))
					{
						if (hitInfo.distance > useCurvesHeight)
						{
							col.height = orgColHight - jumpHeight;			// 調整されたコライダーの高さ
							float adjCenterY = orgVectColCenter.y + jumpHeight;
							col.center = new Vector3(0, adjCenterY, 0);	// 調整されたコライダーのセンター
						}
						else{
							// 閾値よりも低い時には初期値に戻す（念のため）					
							resetCollider();
						}
					}
				}
				// Jump bool値をリセットする（ループしないようにする）				
				anim.SetBool("Jump", false);
			}
		}
		// IDLE中の処理
		// 当前基础层为 idleState 时
		else if (currentBaseState.fullPathHash == idleState)
		{
			//如果您要调整曲线上的碰撞器，请重置它以防万一
			if(useCurves){
				resetCollider();
			}
			// 输入空格键，它将处于休息状态
			if (Input.GetButtonDown("Jump")) {
				anim.SetBool("Rest", true);
			}
		}
		// REST中の処理
		// 当前基础层为restState时
		else if (currentBaseState.fullPathHash == restState)
		{
			//cameraObject.SendMessage("setCameraPositionFrontView");		// 将相机切换到正面
			// ステートが遷移中でない場合、Rest bool値をリセットする（ループしないようにする）
			if(!anim.IsInTransition(0))
			{
				anim.SetBool("Rest", false);
			}
		}
	}

	/*
	void OnGUI()
	{
		GUI.Box(new Rect(Screen.width -260, 10 ,250 ,150), "Interaction");
		GUI.Label(new Rect(Screen.width -245,30,250,30),"Up/Down Arrow : Go Forwald/Go Back");
		GUI.Label(new Rect(Screen.width -245,50,250,30),"Left/Right Arrow : Turn Left/Turn Right");
		GUI.Label(new Rect(Screen.width -245,70,250,30),"Hit Space key while Running : Jump");
		GUI.Label(new Rect(Screen.width -245,90,250,30),"Hit Spase key while Stopping : Rest");
		GUI.Label(new Rect(Screen.width -245,110,250,30),"Left Control : Front Camera");
		GUI.Label(new Rect(Screen.width -245,130,250,30),"Alt : LookAt Camera");
	}
	*/
	
	// キャラクターのコライダーサイズのリセット関数
	void resetCollider()
	{
	// コンポーネントのHeight、Centerの初期値を戻す
		col.height = orgColHight;
		col.center = orgVectColCenter;
	}
}
