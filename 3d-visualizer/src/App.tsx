import { useState, useRef, useMemo, forwardRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, QuadraticBezierLine, Line } from '@react-three/drei'
import './App.css'

const xParam = 10
const yParam = 28
const zParam = 8 / 3

function randomColor() {
  // Generate a random integer between 0 and 255 for each color component
  const red = Math.floor(Math.random() * 256);
  const green = Math.floor(Math.random() * 256);
  const blue = Math.floor(Math.random() * 256);
  
  // Convert each component to hexadecimal, ensuring it is two digits
  const redHex = red.toString(16).padStart(2, '0');
  const greenHex = green.toString(16).padStart(2, '0');
  const blueHex = blue.toString(16).padStart(2, '0');
  
  // Combine the components into a single hex color string
  return `#${redHex}${greenHex}${blueHex}`;
}


function Tracer({ targetRef }) {
  const lineRef = useRef()
  const [points, setPoints] = useState<number[][]>([[0, 0, 0]])
  const color = useMemo(() => randomColor(), [])

  useFrame(() => {
    if (targetRef.current) {
      setPoints((prev) => [...prev, targetRef.current.position.clone()])
    }
  })

  return (
    <Line
      points={points}
      color={color}
    />
  )
}

const Box = forwardRef((props: unknown, ref: any) => {
  // const ref = useRef<unknown>(null)
  useFrame((state, delta) => {
    if (!ref.current) return
    const { x, y, z } = ref.current.position
    const xDelta = xParam * (y - x)
    const yDelta = x * (yParam - z) - y
    const zDelta = x * y - zParam * z

    ref.current.position.x += (xDelta * delta) / 1
    ref.current.position.y += (yDelta * delta) / 1
    ref.current.position.z += (zDelta * delta) / 1
  })
  // Return the view, these are regular Threejs elements expressed in JSX
  return (
    <mesh {...props} ref={ref}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color={'orange'} />
    </mesh>
  )
})

function CameraLogger() {
  const { camera } = useThree()

  useFrame(() => {
    console.log('Camera Position:', camera.position)
    console.log('Camera Rotation:', camera.rotation)
  })

  return null
}

function App() {
  const [count, setCount] = useState(0)
  const boxRefOne = useRef()
  const boxRefTwo = useRef()

  return (
    <Canvas camera={{ position: [-8, 15, -66], fov: 50 }}>
      <ambientLight intensity={Math.PI / 2} />
      <spotLight
        position={[10, 10, 10]}
        angle={0.15}
        penumbra={1}
        decay={0}
        intensity={Math.PI}
      />
      <pointLight position={[-10, -10, -10]} decay={0} intensity={Math.PI} />
      <Box position={[-1.2, 0, 0]} ref={boxRefOne} /> 
      <Box position={[-1.2000001, 0, 0]} ref={boxRefTwo} /> 
      <Tracer targetRef={boxRefOne} />
      <Tracer targetRef={boxRefTwo} />
      <gridHelper args={[500, 1000]} />
      {/* <CameraLogger /> */}
      <OrbitControls />
    </Canvas>
  )
}

export default App
