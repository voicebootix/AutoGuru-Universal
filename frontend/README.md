# AutoGuru Universal Frontend

A modern, responsive React frontend for the AutoGuru Universal social media automation platform. Built with Material-UI, Zustand for state management, and real-time WebSocket integration.

## 🚀 Features

### Universal Design
- **Business Niche Agnostic**: Works for any business type (fitness, education, consulting, creative, etc.)
- **AI-Powered Insights**: Dynamic content and analytics based on your specific business
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices

### Core Features
- **📊 Dashboard**: Real-time overview of performance metrics and recent activity
- **📈 Analytics**: Comprehensive analytics with interactive charts and AI insights
- **📝 Content Management**: Create, schedule, and manage content across all platforms
- **🔗 Platform Integration**: Connect and manage social media platforms with OAuth
- **⚙️ Background Tasks**: Monitor and manage automated background processes
- **⚙️ Settings**: User profile, API keys, and system configuration
- **🆘 Support**: Help center, FAQs, and feedback submission

### Technical Features
- **Real-time Updates**: WebSocket integration for live task and analytics updates
- **State Management**: Zustand for efficient global state management
- **API Integration**: Full integration with AutoGuru Universal backend
- **Error Handling**: Comprehensive error handling and user feedback
- **Loading States**: Smooth loading experiences with skeleton screens
- **Responsive UI**: Material-UI components for consistent design

## 🏗️ Architecture

```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   ├── features/           # Feature-specific components
│   │   ├── analytics/      # Analytics dashboard
│   │   ├── content/        # Content management
│   │   ├── dashboard/      # Main dashboard
│   │   ├── platforms/      # Platform connections
│   │   ├── settings/       # User settings
│   │   ├── support/        # Support & help
│   │   └── tasks/          # Background tasks
│   ├── services/           # API service modules
│   │   ├── api.js         # Base API configuration
│   │   ├── analytics.js   # Analytics API calls
│   │   ├── content.js     # Content API calls
│   │   ├── platforms.js   # Platform API calls
│   │   ├── settings.js    # Settings API calls
│   │   ├── support.js     # Support API calls
│   │   └── tasks.js       # Tasks API calls
│   ├── store/             # Zustand state stores
│   │   ├── analyticsStore.js
│   │   ├── contentStore.js
│   │   ├── platformStore.js
│   │   ├── settingsStore.js
│   │   ├── supportStore.js
│   │   └── tasksStore.js
│   ├── App.jsx            # Main app component
│   └── main.jsx           # App entry point
├── package.json
└── vite.config.js
```

## 🛠️ Tech Stack

- **React 18**: Modern React with hooks and functional components
- **Material-UI (MUI)**: Professional UI component library
- **Zustand**: Lightweight state management
- **React Router**: Client-side routing
- **Axios**: HTTP client for API calls
- **Recharts**: Interactive charts and data visualization
- **Vite**: Fast build tool and development server

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AutoGuru-Universal/frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   Create a `.env` file in the frontend directory:
   ```env
   VITE_API_URL=http://localhost:8000
   ```

4. **Start development server**
   ```bash
   npm run dev
   ```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL | `http://localhost:8000` |

### API Configuration

The frontend automatically configures API calls based on your environment:

- **Development**: Connects to local backend
- **Production**: Connects to deployed backend via `VITE_API_URL`

## 🎯 Usage

### Dashboard
- View real-time performance metrics
- Monitor recent activity and top-performing content
- Quick access to key features

### Analytics
- Interactive charts showing engagement trends
- Platform-specific performance data
- AI-generated insights and recommendations
- Export functionality for data analysis

### Content Management
- Create content with AI assistance
- Schedule posts across multiple platforms
- Track content performance and history
- Bulk operations for efficiency

### Platform Integration
- Connect social media accounts via OAuth
- Monitor connection status and token validity
- Refresh expired tokens automatically
- Platform-specific analytics and insights

### Background Tasks
- Monitor automated processes in real-time
- View task progress and completion status
- Manage task scheduling and priorities
- Real-time updates via WebSocket

### Settings
- Manage user profile and preferences
- Generate and manage API keys
- Configure notification preferences
- System settings and optimizations

### Support
- Comprehensive FAQ section
- Multiple contact methods (email, chat, phone)
- Feedback submission system
- Helpful resources and documentation

## 🔌 API Integration

The frontend integrates with the AutoGuru Universal backend through:

### Service Modules
- **Base API**: Configured Axios instance with authentication
- **Feature Services**: Dedicated modules for each feature area
- **Error Handling**: Global error handling and user feedback
- **Loading States**: Consistent loading experiences

### State Management
- **Zustand Stores**: Feature-specific state management
- **Real-time Updates**: WebSocket integration for live data
- **Caching**: Efficient data caching and updates

### Authentication
- **Token Management**: Automatic token handling
- **OAuth Integration**: Social media platform authentication
- **Session Management**: Secure session handling

## 🎨 UI/UX Features

### Design System
- **Material Design**: Consistent with Google's Material Design
- **Responsive Layout**: Works on all screen sizes
- **Dark/Light Mode**: Theme support (ready for implementation)
- **Accessibility**: WCAG compliant components

### User Experience
- **Loading States**: Skeleton screens and progress indicators
- **Error Handling**: User-friendly error messages
- **Success Feedback**: Confirmation dialogs and notifications
- **Intuitive Navigation**: Clear navigation and breadcrumbs

## 🚀 Deployment

### Build for Production
```bash
npm run build
```

### Deploy to Render
1. Connect your repository to Render
2. Set environment variables in Render dashboard
3. Deploy automatically on push to main branch

### Environment Variables for Production
```env
VITE_API_URL=https://your-backend-url.onrender.com
```

## 🧪 Testing

### Run Tests
```bash
npm test
```

### Test Coverage
```bash
npm run test:coverage
```

## 📊 Performance

### Optimization Features
- **Code Splitting**: Automatic route-based code splitting
- **Lazy Loading**: Components loaded on demand
- **Image Optimization**: Optimized images and assets
- **Bundle Analysis**: Built-in bundle analysis tools

### Monitoring
- **Performance Metrics**: Core Web Vitals tracking
- **Error Tracking**: Automatic error reporting
- **Analytics**: User behavior and performance analytics

## 🔒 Security

### Security Features
- **HTTPS Only**: Secure connections in production
- **CORS Configuration**: Proper cross-origin resource sharing
- **Input Validation**: Client-side validation
- **XSS Protection**: Built-in XSS protection

### Best Practices
- **Environment Variables**: Secure configuration management
- **Token Security**: Secure token storage and handling
- **API Security**: Proper API authentication and authorization

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Standards
- **ESLint**: Code linting and formatting
- **Prettier**: Code formatting
- **TypeScript**: Type safety (optional)
- **Component Structure**: Consistent component organization

## 📚 Documentation

### Additional Resources
- [Material-UI Documentation](https://mui.com/)
- [React Documentation](https://reactjs.org/)
- [Zustand Documentation](https://github.com/pmndrs/zustand)
- [Vite Documentation](https://vitejs.dev/)

### API Documentation
- Backend API documentation available at `/docs` endpoint
- OpenAPI/Swagger specification included

## 🆘 Support

### Getting Help
- **Documentation**: Comprehensive documentation in this README
- **FAQ**: Built-in FAQ section in the app
- **Support Team**: Contact via email, chat, or phone
- **Community**: GitHub issues and discussions

### Common Issues
- **CORS Errors**: Ensure backend CORS is configured correctly
- **API Connection**: Verify `VITE_API_URL` is set correctly
- **Build Issues**: Clear node_modules and reinstall dependencies

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Material-UI team for the excellent component library
- React team for the amazing framework
- Zustand team for lightweight state management
- All contributors to the AutoGuru Universal project 