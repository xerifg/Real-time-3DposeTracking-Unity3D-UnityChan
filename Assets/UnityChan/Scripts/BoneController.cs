using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
public class BoneController : MonoBehaviour
{
	[SerializeField] Animator animator;
	[SerializeField, Range(10, 120)] float FrameRate;
	[SerializeField] GameObject BoneRoot;
	[SerializeField] string Data_Path;
	[SerializeField] string File_Name;
	[SerializeField] int Data_Size;
	public List<Transform> BoneList = new List<Transform>();
	Vector3[] points = new Vector3[17];
	Vector3[] DefaultNormalizeBone = new Vector3[12];
	Vector3[] NormalizeBone = new Vector3[12];
	Vector3[] LerpedNormalizeBone = new Vector3[12];

	Quaternion[] DefaultBoneRot = new Quaternion[17];
	Quaternion[] DefaultBoneLocalRot = new Quaternion[17];
	Vector3[] DefaultXAxis = new Vector3[17];
	Vector3[] DefaultYAxis = new Vector3[17];
	Vector3[] DefaultZAxis = new Vector3[17];

	float Timer;
	int[,] joints = new int[,]
	{ { 0, 1 }, { 1, 2 }, { 2, 3 }, { 0, 4 }, { 4, 5 }, { 5, 6 }, { 0, 7 }, { 7, 8 }, { 8, 9 }, { 9, 10 }, { 8, 11 }, { 11, 12 }, { 12, 13 }, { 8, 14 }, { 14, 15 }, { 15, 16 }
	};
	int[,] BoneJoint = new int[,]
	{ { 0, 2 }, { 2, 3 }, { 0, 5 }, { 5, 6 }, { 0, 7 }, { 7, 8 }, { 8, 9 }, { 9, 10 }, { 9, 12 }, { 12, 13 }, { 9, 15 }, { 15, 16 }
	};
	int NowFrame = 0;
	void Start()
	{
		GetBones();
		PointUpdate();
	}

	void Update()
	{
		PointUpdateByTime();  // obtain a new data every frame rate
		SetBoneRot();
	}
	void GetBones()
	{
		BoneList.Add(animator.GetBoneTransform(HumanBodyBones.Hips));  // get points location
		BoneList.Add(animator.GetBoneTransform(HumanBodyBones.LeftUpperLeg));
		BoneList.Add(animator.GetBoneTransform(HumanBodyBones.LeftLowerLeg));
		BoneList.Add(animator.GetBoneTransform(HumanBodyBones.LeftFoot));
		BoneList.Add(animator.GetBoneTransform(HumanBodyBones.RightUpperLeg));
		BoneList.Add(animator.GetBoneTransform(HumanBodyBones.RightLowerLeg));
		BoneList.Add(animator.GetBoneTransform(HumanBodyBones.RightFoot));
		BoneList.Add(animator.GetBoneTransform(HumanBodyBones.Spine));
		BoneList.Add(animator.GetBoneTransform(HumanBodyBones.Chest));
		BoneList.Add(animator.GetBoneTransform(HumanBodyBones.Neck));
		BoneList.Add(animator.GetBoneTransform(HumanBodyBones.Head));
		BoneList.Add(animator.GetBoneTransform(HumanBodyBones.RightUpperArm));
		BoneList.Add(animator.GetBoneTransform(HumanBodyBones.RightLowerArm));
		BoneList.Add(animator.GetBoneTransform(HumanBodyBones.RightHand));
		BoneList.Add(animator.GetBoneTransform(HumanBodyBones.LeftUpperArm));
		BoneList.Add(animator.GetBoneTransform(HumanBodyBones.LeftLowerArm));
		BoneList.Add(animator.GetBoneTransform(HumanBodyBones.LeftHand));

		for (int i = 0; i < 17; i++)
		{
			var rootT = animator.GetBoneTransform(HumanBodyBones.Hips).root;  //get the animator's transform
			DefaultBoneRot[i] = BoneList[i].rotation;  // get all points Initially absolute quaternion angle
			DefaultBoneLocalRot[i] = BoneList[i].localRotation;   // get all points Initially relatively quaternion angle
			DefaultXAxis[i] = new Vector3(
				Vector3.Dot(BoneList[i].right, rootT.right),  //calculate Included angle
				Vector3.Dot(BoneList[i].up, rootT.right),
				Vector3.Dot(BoneList[i].forward, rootT.right)
			);
			DefaultYAxis[i] = new Vector3(
				Vector3.Dot(BoneList[i].right, rootT.up),
				Vector3.Dot(BoneList[i].up, rootT.up),
				Vector3.Dot(BoneList[i].forward, rootT.up)
			);
			DefaultZAxis[i] = new Vector3(
				Vector3.Dot(BoneList[i].right, rootT.forward),
				Vector3.Dot(BoneList[i].up, rootT.forward),
				Vector3.Dot(BoneList[i].forward, rootT.forward)
			);
		}
		for (int i = 0; i < 12; i++)
		{
			DefaultNormalizeBone[i] = (BoneList[BoneJoint[i, 1]].position - BoneList[BoneJoint[i, 0]].position).normalized;  //Normalized skeleton length
		}
	}
	void PointUpdate()
	{
		if (NowFrame < Data_Size)
		{
			StreamReader fi = new StreamReader(Application.dataPath + Data_Path + File_Name + NowFrame.ToString() + ".txt");
			//NowFrame++;
			string all = fi.ReadToEnd();
			if (all != "0")
			{
				string[] axis = all.Split(']');
				float[] x = axis[0].Replace("[", "").Replace("\r\n", "").Replace("\n", "").Split(' ')
						.Where(s => s != "").Select(f => float.Parse(f)).ToArray();
				float[] y = axis[2].Replace("[", "").Replace("\r\n", "").Replace("\n", "").Split(' ')
						.Where(s => s != "").Select(f => float.Parse(f)).ToArray();
				float[] z = axis[1].Replace("[", "").Replace("\r\n", "").Replace("\n", "").Split(' ')
						.Where(s => s != "").Select(f => float.Parse(f)).ToArray();
				for (int i = 0; i < 17; i++)
				{
					points[i] = new Vector3(x[i], y[i], -z[i]);  //get 17 joint points coordination 
				}
				for (int i = 0; i < 12; i++)
				{
					NormalizeBone[i] = (points[BoneJoint[i, 1]] - points[BoneJoint[i, 0]]).normalized;  //Normalized skeleton length
				}
			}
			else
			{
				Debug.Log("All Data 0");
			}
		}
	}
	void PointUpdateByTime()
	{
		Timer += Time.deltaTime;
		if (Timer > (1 / FrameRate))
		{
			Timer = 0;
			PointUpdate();
		}
	}
	Quaternion GetBoneRot(int jointNum)
	{
		Quaternion target = Quaternion.FromToRotation(DefaultNormalizeBone[jointNum], LerpedNormalizeBone[jointNum]);  // get quaternion from default to target
		return target;
	}
	void SetBoneRot()
	{
		for (int i = 0; i < 12; i++)
		{
			LerpedNormalizeBone[i] = Vector3.Slerp(LerpedNormalizeBone[i], NormalizeBone[i], 0.1f);  //Spherical interpolation
		}
		/*
		if (Math.Abs(points[0].x) < 1000 && Math.Abs(points[0].y) < 1000 && Math.Abs(points[0].z) < 1000)
		{
			BoneList[0].position = Vector3.Lerp(BoneList[0].position, points[0] * 0.001f + Vector3.up * 0.8f, 0.1f);  //Linear interpolation
			Vector3 hipRot = (NormalizeBone[0] + NormalizeBone[2] + NormalizeBone[4]).normalized;
			BoneRoot.transform.forward = Vector3.Lerp(BoneRoot.transform.forward, new Vector3(hipRot.x, 0, hipRot.z), 0.1f);
		}
		*/
		int j = 0;
		for (int i = 1; i < 17; i++)
		{
			if (i != 3 && i != 6 && i != 13 && i != 16)
			{
				float angle;
				Vector3 axis;
				GetBoneRot(j).ToAngleAxis(out angle, out axis); //将Quaternion实例转换为角轴表示;参数angle为旋转角；参数axis为轴向量;in world axis

				Vector3 axisInLocalCoordinate = axis.x * DefaultXAxis[i] + axis.y * DefaultYAxis[i] + axis.z * DefaultZAxis[i];  // from world axis to local axis

				Quaternion modifiedRotation = Quaternion.AngleAxis(angle, axisInLocalCoordinate); // from angle axis to quaternion

				BoneList[i].localRotation = Quaternion.Lerp(BoneList[i].localRotation, DefaultBoneLocalRot[i] * modifiedRotation, 0.1f);  // rotate bones by localrotation
				j++;
			}
		}
		for (int i = 0; i < 16; i++)
		{
			DrawLine(points[joints[i, 0]] * 0.001f + new Vector3(-1, 0.8f, 0), points[joints[i, 1]] * 0.001f + new Vector3(-1, 0.8f, 0), Color.blue);
			DrawRay(points[joints[i, 0]] * 0.001f + new Vector3(-1, 0.8f, 0), BoneList[i].right * 0.1f, Color.magenta);
			DrawRay(points[joints[i, 0]] * 0.001f + new Vector3(-1, 0.8f, 0), BoneList[i].up * 0.1f, Color.green);
			DrawRay(points[joints[i, 0]] * 0.001f + new Vector3(-1, 0.8f, 0), BoneList[i].forward * 0.1f, Color.cyan);
		}
		for (int i = 0; i < 12; i++)
		{
			//DrawRay(points[BoneJoint[i, 0]] * 0.001f + new Vector3(1, 0.8f, 0), NormalizeBone[i] * 0.25f, Color.green);
		}
	}
	void DrawLine(Vector3 s, Vector3 e, Color c)
	{
		Debug.DrawLine(s, e, c);
	}
	void DrawRay(Vector3 s, Vector3 d, Color c)
	{
		Debug.DrawRay(s, d, c);
	}
}
enum PointsNum
{
	Hips,
	RightUpperLeg,
	RightLowerLeg,
	RightFoot,
	LeftUpperLeg,
	LeftLowerLeg,
	LeftFoot,
	Spine,
	Chest,
	Neck,
	Head,
	LeftUpperArm,
	LeftLowerArm,
	LeftHand,
	RightUpperArm,
	RightLowerArm,
	RightHand
}