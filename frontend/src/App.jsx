import "./App.css";

function App() {
  return (
    <div className="container">

<h1>
  🏏 Hindi Sports Review
  <br />
  Sentiment Analyzer
</h1>

<p>
  Detect the sentiment of Hindi sports comments using AI-powered Natural Language Processing.
</p>

      <textarea
        placeholder="Enter your Hindi sports comment here..."
      />

      <button>
        Analyze Sentiment
      </button>

    </div>
  );
}

export default App;