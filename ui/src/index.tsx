import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { App } from './app/components/App';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement)
   .render(
      <BrowserRouter>
         <App/>
      </BrowserRouter>
   )