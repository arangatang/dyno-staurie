import React, { useState, useEffect } from "react"

function App() {
  const [data, setData] = useState([{}])
  useEffect(() => {
    fetch("/members").then(
      res => res.json()
    ).then(
      data => {
        setData(data)
        console.log(data)
      }
    )
  }, [])

  // incoming json will have the following structure
  /**
   * {
   *    'text': ...,
   *    'options': {
   *        'option_id': 'text',
   *        },
   *    'image': <url to image>
   * }
   */
  //data.text = "test"
  //data.options = {"option1": "the first option"}
  // data.image = 'https://www.w3schools.com/images/htmlvideoad_footer.png'
   
  return (
    <div>
      {(typeof data.text === 'undefined') ? (
        <p>Loading...</p>
      ) : (
        <div>
          <img
            src={data.image}
            alt={data.text}
          />    
          {

            Object.entries(data.options).map(
            ([option_id, text]) => (
              <button key={option_id} id={option_id}>{text}</button>
            ))
          } 
        </div>

      )}
    </div>
  )
}

export default App